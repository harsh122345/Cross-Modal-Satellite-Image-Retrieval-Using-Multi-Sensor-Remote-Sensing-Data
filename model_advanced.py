import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

# Global device setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FallbackVisualEncoder(nn.Module):
    def __init__(self, output_dim=384):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, output_dim)
        
        # Activation and gradient storage for Grad-CAM
        self.gradients = None
        self.activations = None
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        
        self.activations = x
        if x.requires_grad:
            x.register_hook(self._save_gradient)
            
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
        
    def _save_gradient(self, grad):
        self.gradients = grad


class FallbackTextEncoder(nn.Module):
    def __init__(self, output_dim=512, vocab_size=1000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64, padding_idx=0)
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        self.vocab = {"<pad>": 0}
        
    def _tokenize(self, text):
        words = text.lower().replace(".", "").replace(",", "").split()
        ids = []
        for w in words:
            if w not in self.vocab:
                if len(self.vocab) < 1000:
                    self.vocab[w] = len(self.vocab)
                else:
                    continue
            ids.append(self.vocab[w])
        if len(ids) < 20:
            ids += [0] * (20 - len(ids))
        else:
            ids = ids[:20]
        return torch.tensor(ids, dtype=torch.long)
        
    def forward(self, texts):
        token_tensors = [self._tokenize(t) for t in texts]
        batch_tokens = torch.stack(token_tensors).to(DEVICE)
        embedded = self.embedding(batch_tokens)
        pooled = torch.mean(embedded, dim=1)
        return self.fc(pooled)


class CLIPWrapper:
    def __init__(self):
        self.use_fallback = False
        try:
            from transformers import CLIPProcessor, CLIPModel
            print("Attempting to load CLIP model...")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=False)
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=False)
            self.model.to(DEVICE)
            self.model.eval()
            print("Successfully loaded pre-trained CLIP model.")
        except Exception as e:
            print(f"CLIP load failed ({e}). Using lightweight custom PyTorch fallback.")
            self.use_fallback = True
            self.image_encoder = FallbackVisualEncoder(output_dim=512).to(DEVICE)
            self.text_encoder = FallbackTextEncoder(output_dim=512).to(DEVICE)
            
    def encode_image(self, pil_images):
        if self.use_fallback:
            tensors = [self._preprocess_image(img) for img in pil_images]
            batch_tensor = torch.stack(tensors).to(DEVICE)
            self.image_encoder.eval()
            with torch.no_grad():
                embeddings = self.image_encoder(batch_tensor)
                embeddings = F.normalize(embeddings, p=2, dim=-1)
            return embeddings.cpu().numpy()
        else:
            inputs = self.processor(images=pil_images, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                if not isinstance(image_features, torch.Tensor):
                    if hasattr(image_features, "pooler_output") and image_features.pooler_output is not None:
                        image_features = image_features.pooler_output
                    elif hasattr(image_features, "last_hidden_state") and image_features.last_hidden_state is not None:
                        image_features = image_features.last_hidden_state[:, 0, :]
                    elif isinstance(image_features, (list, tuple)):
                        image_features = image_features[0]
                image_features = F.normalize(image_features, p=2, dim=-1)
            return image_features.cpu().numpy()
            
    def encode_text(self, texts):
        if self.use_fallback:
            self.text_encoder.eval()
            with torch.no_grad():
                embeddings = self.text_encoder(texts)
                embeddings = F.normalize(embeddings, p=2, dim=-1)
            return embeddings.cpu().numpy()
        else:
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                if not isinstance(text_features, torch.Tensor):
                    if hasattr(text_features, "pooler_output") and text_features.pooler_output is not None:
                        text_features = text_features.pooler_output
                    elif isinstance(text_features, (list, tuple)):
                        text_features = text_features[0]
                text_features = F.normalize(text_features, p=2, dim=-1)
            return text_features.cpu().numpy()
            
    def _preprocess_image(self, pil_img):
        img = pil_img.resize((64, 64))
        arr = np.array(img, dtype=np.float32) / 255.0
        if len(arr.shape) == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        if len(arr.shape) == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        arr = np.transpose(arr, (2, 0, 1))
        return torch.tensor(arr)


class Dinov2Wrapper:
    def __init__(self):
        self.use_fallback = False
        try:
            from transformers import AutoImageProcessor, Dinov2Model
            print("Attempting to load DINOv2 model...")
            self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small", local_files_only=False)
            self.model = Dinov2Model.from_pretrained("facebook/dinov2-small", local_files_only=False)
            self.model.to(DEVICE)
            self.model.eval()
            print("Successfully loaded pre-trained DINOv2 model.")
        except Exception as e:
            print(f"DINOv2 load failed ({e}). Using lightweight custom PyTorch fallback.")
            self.use_fallback = True
            self.image_encoder = FallbackVisualEncoder(output_dim=384).to(DEVICE)
            
    def extract_features(self, pil_images):
        if self.use_fallback:
            tensors = [self._preprocess_image(img) for img in pil_images]
            batch_tensor = torch.stack(tensors).to(DEVICE)
            self.image_encoder.eval()
            with torch.no_grad():
                features = self.image_encoder(batch_tensor)
            return features.cpu().numpy()
        else:
            inputs = self.processor(images=pil_images, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = self.model(**inputs)
                features = outputs.last_hidden_state[:, 0, :]
            return features.cpu().numpy()
            
    def _preprocess_image(self, pil_img):
        img = pil_img.resize((64, 64))
        arr = np.array(img, dtype=np.float32) / 255.0
        if len(arr.shape) == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        if len(arr.shape) == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        arr = np.transpose(arr, (2, 0, 1))
        return torch.tensor(arr)


class MultiModalTransformer(nn.Module):
    def __init__(self, opt_dim=17, sar_dim=6, txt_dim=20, embed_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.opt_proj = nn.Linear(opt_dim, embed_dim)
        self.sar_proj = nn.Linear(sar_dim, embed_dim)
        self.txt_proj = nn.Linear(txt_dim, embed_dim)
        
        self.pos_embed = nn.Parameter(torch.randn(3, 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4, 
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, opt_feat, sar_feat, txt_feat):
        # Input shape: (B, dim)
        opt_p = self.opt_proj(opt_feat).unsqueeze(0)  # (1, B, embed_dim)
        sar_p = self.sar_proj(sar_feat).unsqueeze(0)  # (1, B, embed_dim)
        txt_p = self.txt_proj(txt_feat).unsqueeze(0)  # (1, B, embed_dim)
        
        x = torch.cat([opt_p, sar_p, txt_p], dim=0)    # (3, B, embed_dim)
        x = x + self.pos_embed
        
        fused = self.transformer(x)                    # (3, B, embed_dim)
        fused_mean = torch.mean(fused, dim=0)          # (B, embed_dim)
        
        output = self.fc(fused_mean)
        return F.normalize(output, p=2, dim=-1)


class GCNRefiner(nn.Module):
    def __init__(self, in_features=64, hidden_features=64, out_features=64):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        
    def forward(self, x, adj_matrix):
        # x: (N, in_features)
        # adj_matrix: (N, N)
        N = x.size(0)
        I = torch.eye(N, device=x.device)
        A_tilde = adj_matrix + I
        
        deg = torch.sum(A_tilde, dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        
        A_norm = torch.mm(torch.mm(D_inv_sqrt, A_tilde), D_inv_sqrt)
        
        h1 = F.relu(self.fc1(torch.mm(A_norm, x)))
        h2 = self.fc2(torch.mm(A_norm, h1))
        
        return F.normalize(h2, p=2, dim=-1)


def generate_saliency_heatmap(pil_image, wrapper, query_embedding=None):
    """
    Generates a 2D saliency heatmap (64x64) representing model attention.
    If query_embedding is provided, it targets similarity.
    Uses the wrapper's fallback model or a custom CNN.
    """
    encoder = getattr(wrapper, "image_encoder", None)
    if encoder is None:
        encoder = FallbackVisualEncoder(output_dim=384).to(DEVICE)
        
    # Ensure encoder is in train mode so activations can receive gradients
    encoder.train()
    
    # Preprocess image
    img_resized = pil_image.resize((64, 64))
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    if len(arr.shape) == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if len(arr.shape) == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    arr = np.transpose(arr, (2, 0, 1))
    input_tensor = torch.tensor(arr).unsqueeze(0).to(DEVICE).requires_grad_(True)
    
    encoder.zero_grad()
    features = encoder(input_tensor)
    
    if query_embedding is not None:
        target_t = torch.tensor(query_embedding).to(DEVICE).reshape(1, -1)
        if features.shape[1] != target_t.shape[1]:
            proj = nn.Linear(features.shape[1], target_t.shape[1]).to(DEVICE)
            loss = torch.sum(proj(features) * target_t)
        else:
            loss = torch.sum(features * target_t)
    else:
        loss = torch.sum(features)
        
    loss.backward()
    
    gradients = encoder.gradients
    activations = encoder.activations
    
    if gradients is not None and activations is not None:
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        weighted_act = activations.clone()
        for i in range(weighted_act.shape[1]):
            weighted_act[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(weighted_act, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        max_val = torch.max(heatmap)
        if max_val > 0:
            heatmap /= max_val
        heatmap_np = heatmap.cpu().detach().numpy()
        
        import scipy.ndimage as ndimage
        zoom_factor = 64.0 / heatmap_np.shape[0]
        heatmap_resized = ndimage.zoom(heatmap_np, zoom_factor, order=1)
        heatmap_resized = np.clip(heatmap_resized, 0, 1)
        return heatmap_resized
    else:
        # Gradients failed fallback
        gray = 0.299 * arr[0] + 0.587 * arr[1] + 0.114 * arr[2]
        dx = np.abs(gray[:-1, 1:] - gray[:-1, :-1])
        dy = np.abs(gray[1:, :-1] - gray[:-1, :-1])
        grad = np.zeros_like(gray)
        grad[:-1, :-1] = np.sqrt(dx**2 + dy**2)
        grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
        import scipy.ndimage as ndimage
        grad_smooth = ndimage.gaussian_filter(grad, sigma=2.0)
        grad_smooth = (grad_smooth - grad_smooth.min()) / (grad_smooth.max() - grad_smooth.min() + 1e-8)
        return grad_smooth
