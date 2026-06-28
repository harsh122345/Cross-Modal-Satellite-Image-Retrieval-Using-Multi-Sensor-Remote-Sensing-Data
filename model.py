import numpy as np

class MultiModalProjectionBranch:
    """
    A 2-layer Neural Network MLP Projection Branch:
    Input -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> Output (to be normalized)
    Implemented in pure NumPy with Adam Optimizer states.
    """
    def __init__(self, input_dim, hidden_dim, embed_dim, seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        # He (Kaiming) initialization for W1, Xavier initialization for W2
        self.W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim), dtype=np.float32)
        
        self.W2 = np.random.randn(hidden_dim, embed_dim).astype(np.float32) * np.sqrt(1.0 / hidden_dim)
        self.b2 = np.zeros((1, embed_dim), dtype=np.float32)
        
        # Adam Optimizer states (momentum m, scaling v)
        self.m_W1, self.v_W1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        self.m_b1, self.v_b1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        self.m_W2, self.v_W2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        self.m_b2, self.v_b2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
        
        # Step counter
        self.t = 0
        
        # Saved activations for backprop
        self.x = None
        self.h = None
        self.u = None
        self.z = None

    def forward(self, x):
        """
        Computes forward pass: Linear -> ReLU -> Linear.
        Does NOT normalize here; normalization is handled in the joint class.
        """
        self.x = x
        # Layer 1
        self.h = np.dot(x, self.W1) + self.b1
        self.h = np.maximum(0.0, self.h) # ReLU
        # Layer 2
        self.u = np.dot(self.h, self.W2) + self.b2
        return self.u

    def backward(self, g_u):
        """
        Backward pass of the MLP branch given the gradient w.r.t the unnormalized outputs self.u.
        Returns:
            g_W1, g_b1, g_W2, g_b2
        """
        # Gradients for Layer 2
        g_W2 = np.dot(self.h.T, g_u)
        g_b2 = np.sum(g_u, axis=0, keepdims=True)
        
        # Backprop through Layer 2 to hidden activation
        g_h = np.dot(g_u, self.W2.T)
        
        # Backprop through ReLU
        g_pre = g_h * (self.h > 0.0)
        
        # Gradients for Layer 1
        g_W1 = np.dot(self.x.T, g_pre)
        g_b1 = np.sum(g_pre, axis=0, keepdims=True)
        
        return g_W1, g_b1, g_W2, g_b2

    def update_parameters(self, grads, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Updates weights and biases using Adam optimizer.
        """
        g_W1, g_b1, g_W2, g_b2 = grads
        self.t += 1
        
        # Update Adam states and update weights
        for param, grad, m, v in [
            (self.W1, g_W1, self.m_W1, self.v_W1),
            (self.b1, g_b1, self.m_b1, self.v_b1),
            (self.W2, g_W2, self.m_W2, self.v_W2),
            (self.b2, g_b2, self.m_b2, self.v_b2)
        ]:
            # Update biased first moment estimate
            np.copyto(m, beta1 * m + (1.0 - beta1) * grad)
            # Update biased second raw moment estimate
            np.copyto(v, beta2 * v + (1.0 - beta2) * (grad ** 2))
            
            # Compute bias-corrected first moment estimate
            m_hat = m / (1.0 - beta1 ** self.t)
            # Compute bias-corrected second raw moment estimate
            v_hat = v / (1.0 - beta2 ** self.t)
            
            # Apply update
            param -= (lr * m_hat) / (np.sqrt(v_hat) + eps)


class CrossModalEmbeddingAligner:
    """
    Symmetric Multi-Modal Alignment Network mapping:
    - Optical RGB Features (17d)
    - SAR Grayscale Features (6d)
    - Text Count Features (20d)
    to a shared, aligned embedding space of dimension embed_dim.
    """
    def __init__(self, dim_opt=17, dim_sar=6, dim_txt=20, hidden_dim=32, embed_dim=16, temperature=0.1, seed=42):
        self.embed_dim = embed_dim
        self.temperature = temperature
        
        # Branches
        self.branch_opt = MultiModalProjectionBranch(dim_opt, hidden_dim, embed_dim, seed=seed)
        self.branch_sar = MultiModalProjectionBranch(dim_sar, hidden_dim, embed_dim, seed=seed + 1)
        self.branch_txt = MultiModalProjectionBranch(dim_txt, hidden_dim, embed_dim, seed=seed + 2)
        
        # Feature normalizers (Means & Stds)
        self.mean_opt, self.std_opt = None, None
        self.mean_sar, self.std_sar = None, None
        self.mean_txt, self.std_txt = None, None
        
    def fit_normalizers(self, X_opt, X_sar, X_txt):
        """
        Fits mean and standard deviation parameters for feature normalization.
        """
        self.mean_opt, self.std_opt = np.mean(X_opt, axis=0), np.std(X_opt, axis=0)
        self.mean_sar, self.std_sar = np.mean(X_sar, axis=0), np.std(X_sar, axis=0)
        self.mean_txt, self.std_txt = np.mean(X_txt, axis=0), np.std(X_txt, axis=0)
        
        # Prevent division by zero
        self.std_opt[self.std_opt == 0] = 1.0
        self.std_sar[self.std_sar == 0] = 1.0
        self.std_txt[self.std_txt == 0] = 1.0
        
    def normalize_features(self, X_opt, X_sar, X_txt):
        """
        Applies standard scaler scaling.
        """
        X_opt_norm = (X_opt - self.mean_opt) / self.std_opt
        X_sar_norm = (X_sar - self.mean_sar) / self.std_sar
        X_txt_norm = (X_txt - self.mean_txt) / self.std_txt
        return X_opt_norm, X_sar_norm, X_txt_norm
        
    def forward(self, X_opt, X_sar, X_txt, normalized=True):
        """
        Performs forward pass for all three modalities and returns normalized embeddings.
        """
        if normalized and self.mean_opt is not None:
            X_opt, X_sar, X_txt = self.normalize_features(X_opt, X_sar, X_txt)
            
        u_opt = self.branch_opt.forward(X_opt)
        u_sar = self.branch_sar.forward(X_sar)
        u_txt = self.branch_txt.forward(X_txt)
        
        # Row-wise L2 Normalization
        norm_opt = np.linalg.norm(u_opt, axis=1, keepdims=True)
        norm_sar = np.linalg.norm(u_sar, axis=1, keepdims=True)
        norm_txt = np.linalg.norm(u_txt, axis=1, keepdims=True)
        
        # Avoid zero division
        norm_opt[norm_opt == 0] = 1e-8
        norm_sar[norm_sar == 0] = 1e-8
        norm_txt[norm_txt == 0] = 1e-8
        
        z_opt = u_opt / norm_opt
        z_sar = u_sar / norm_sar
        z_txt = u_txt / norm_txt
        
        # Store for backpropagation
        self.branch_opt.z = z_opt
        self.branch_sar.z = z_sar
        self.branch_txt.z = z_txt
        
        return z_opt, z_sar, z_txt

    def compute_contrastive_loss_and_grads(self, z_a, z_b, u_a, u_b):
        """
        Computes symmetric InfoNCE Contrastive loss and outputs the gradients
        w.r.t the unnormalized inputs u_a and u_b.
        """
        B = z_a.shape[0]
        
        # Cosine similarity matrix (B x B)
        S = np.dot(z_a, z_b.T)
        
        # Softmax probabilities
        exp_row = np.exp(S / self.temperature)
        P_row = exp_row / np.sum(exp_row, axis=1, keepdims=True)
        
        exp_col = np.exp(S / self.temperature)
        P_col = exp_col / np.sum(exp_col, axis=0, keepdims=True)
        
        # Symmetric Loss
        identity = np.eye(B, dtype=np.float32)
        loss_row = -np.mean(np.log(np.diag(P_row) + 1e-8))
        loss_col = -np.mean(np.log(np.diag(P_col) + 1e-8))
        loss = 0.5 * (loss_row + loss_col)
        
        # Gradient w.r.t Similarity matrix S
        g_S = (1.0 / (2.0 * B * self.temperature)) * (P_row + P_col - 2.0 * identity)
        
        # Gradients w.r.t normalized embeddings z_a and z_b
        g_za = np.dot(g_S, z_b)
        g_zb = np.dot(g_S.T, z_a)
        
        # Backprop through L2-normalization for branch A
        norm_a = np.linalg.norm(u_a, axis=1, keepdims=True)
        norm_a[norm_a == 0] = 1e-8
        dot_a = np.sum(z_a * g_za, axis=1, keepdims=True)
        g_ua = (g_za - dot_a * z_a) / norm_a
        
        # Backprop through L2-normalization for branch B
        norm_b = np.linalg.norm(u_b, axis=1, keepdims=True)
        norm_b[norm_b == 0] = 1e-8
        dot_b = np.sum(z_b * g_zb, axis=1, keepdims=True)
        g_ub = (g_zb - dot_b * z_b) / norm_b
        
        return loss, g_ua, g_ub

    def train_step(self, X_opt, X_sar, X_txt, lr=0.001):
        """
        Executes one gradient descent step aligning all three branches pairwise:
        L_total = L(Optical, SAR) + L(Optical, Text) + L(SAR, Text)
        """
        # Forward pass (normalized features)
        z_opt, z_sar, z_txt = self.forward(X_opt, X_sar, X_txt, normalized=True)
        
        # Get unnormalized outputs
        u_opt = self.branch_opt.u
        u_sar = self.branch_sar.u
        u_txt = self.branch_txt.u
        
        # 1. Optical-SAR Pair Alignment
        loss_os, g_u_os_opt, g_u_os_sar = self.compute_contrastive_loss_and_grads(z_opt, z_sar, u_opt, u_sar)
        
        # 2. Optical-Text Pair Alignment
        loss_ot, g_u_ot_opt, g_u_ot_txt = self.compute_contrastive_loss_and_grads(z_opt, z_txt, u_opt, u_txt)
        
        # 3. SAR-Text Pair Alignment
        loss_st, g_u_st_sar, g_u_st_txt = self.compute_contrastive_loss_and_grads(z_sar, z_txt, u_sar, u_txt)
        
        # Accumulate Loss
        total_loss = loss_os + loss_ot + loss_st
        
        # Accumulate unnormalized gradients for each branch
        g_u_opt_total = g_u_os_opt + g_u_ot_opt
        g_u_sar_total = g_u_os_sar + g_u_st_sar
        g_u_txt_total = g_u_ot_txt + g_u_st_txt
        
        # Compute branch parameter gradients
        grads_opt = self.branch_opt.backward(g_u_opt_total)
        grads_sar = self.branch_sar.backward(g_u_sar_total)
        grads_txt = self.branch_txt.backward(g_u_txt_total)
        
        # Update parameters via Adam
        self.branch_opt.update_parameters(grads_opt, lr)
        self.branch_sar.update_parameters(grads_sar, lr)
        self.branch_txt.update_parameters(grads_txt, lr)
        
        return {
            'loss': total_loss,
            'loss_os': loss_os,
            'loss_ot': loss_ot,
            'loss_st': loss_st
        }
