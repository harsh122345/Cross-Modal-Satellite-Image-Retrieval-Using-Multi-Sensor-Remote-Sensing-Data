import unittest
import numpy as np
import torch
from PIL import Image

from dataset import SatelliteDatasetGenerator, build_spatial_graph
from model_advanced import CLIPWrapper, Dinov2Wrapper, MultiModalTransformer, GCNRefiner, generate_saliency_heatmap
from index_search import FaissIndexManager

class TestAdvancedFeatures(unittest.TestCase):
    def setUp(self):
        self.generator = SatelliteDatasetGenerator(size=64)
        self.dataset = self.generator.generate_dataset(num_samples=10, seed=42)
        
    def test_dataset_coordinates(self):
        # Verify that dataset has coordinates and grid coordinates
        for i, patch in enumerate(self.dataset):
            self.assertIn('lat', patch)
            self.assertIn('lon', patch)
            self.assertIn('row', patch)
            self.assertIn('col', patch)
            self.assertIn('id', patch)
            # Bound check: coordinates should be populated
            self.assertNotEqual(patch['lat'], 0.0)
            self.assertNotEqual(patch['lon'], 0.0)
            
    def test_spatial_graph(self):
        G = build_spatial_graph(self.dataset)
        self.assertEqual(len(G.nodes), 10)
        # Should have edges for 4-connectivity
        self.assertTrue(len(G.edges) > 0)
        
    def test_clip_wrapper(self):
        clip = CLIPWrapper()
        # Test visual encoding
        images = [p['optical'] for p in self.dataset[:2]]
        img_embeds = clip.encode_image(images)
        self.assertEqual(img_embeds.shape, (2, 512))
        
        # Test text encoding
        texts = [p['description'] for p in self.dataset[:2]]
        text_embeds = clip.encode_text(texts)
        self.assertEqual(text_embeds.shape, (2, 512))
        
    def test_dinov2_wrapper(self):
        dinov2 = Dinov2Wrapper()
        images = [p['optical'] for p in self.dataset[:2]]
        features = dinov2.extract_features(images)
        self.assertEqual(features.shape, (2, 384))
        
    def test_multimodal_transformer(self):
        opt_dim = 17
        sar_dim = 6
        txt_dim = 20
        model = MultiModalTransformer(opt_dim, sar_dim, txt_dim, embed_dim=64)
        
        opt_feat = torch.randn(4, opt_dim)
        sar_feat = torch.randn(4, sar_dim)
        txt_feat = torch.randn(4, txt_dim)
        
        out = model(opt_feat, sar_feat, txt_feat)
        self.assertEqual(out.shape, (4, 64))
        
        # Test loss and backward pass
        loss = out.sum()
        loss.backward()
        
        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                
    def test_gcn_refiner(self):
        model = GCNRefiner(in_features=64, hidden_features=64, out_features=64)
        x = torch.randn(10, 64)
        adj = torch.zeros(10, 10)
        adj[0, 1] = 1.0
        adj[1, 0] = 1.0
        
        out = model(x, adj)
        self.assertEqual(out.shape, (10, 64))
        
    def test_faiss_index_manager(self):
        manager = FaissIndexManager(dimension=64, metric='cosine')
        embeddings = np.random.randn(10, 64)
        manager.add(embeddings)
        manager.finalize()
        
        # Search
        query = np.random.randn(64)
        indices, distances = manager.search(query, k=3)
        self.assertEqual(len(indices), 3)
        self.assertEqual(len(distances), 3)
        
        # Benchmark
        bench = manager.benchmark_search(query, k=3, runs=10)
        self.assertIn('numpy_latency_ms', bench)
        self.assertIn('faiss_latency_ms', bench)
        
    def test_saliency_heatmap(self):
        dinov2 = Dinov2Wrapper()
        img = self.dataset[0]['optical']
        heatmap = generate_saliency_heatmap(img, dinov2)
        self.assertEqual(heatmap.shape, (64, 64))
        self.assertTrue(heatmap.max() <= 1.0)
        self.assertTrue(heatmap.min() >= 0.0)

if __name__ == '__main__':
    unittest.main()
