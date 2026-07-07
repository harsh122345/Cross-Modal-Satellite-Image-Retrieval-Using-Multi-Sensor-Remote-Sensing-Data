import numpy as np
import time

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS-cpu not available. Falling back to NumPy-based similarity matching.")

class FaissIndexManager:
    """
    Manages indexing and real-time retrieval of high-dimensional multi-modal embeddings.
    Features: FAISS L2 and Cosine Indexing with a robust NumPy-based cosine matching fallback.
    """
    def __init__(self, dimension, metric='cosine'):
        self.dimension = dimension
        self.metric = metric.lower()
        self.index = None
        self.embeddings = []
        self.all_embeddings = None
        
        if FAISS_AVAILABLE:
            if self.metric == 'cosine':
                # Cosine similarity in FAISS is Inner Product (IndexFlatIP) on normalized vectors
                self.index = faiss.IndexFlatIP(self.dimension)
            elif self.metric == 'l2':
                self.index = faiss.IndexFlatL2(self.dimension)
            else:
                self.index = faiss.IndexFlatL2(self.dimension)
        else:
            self.index = None
            
    def add(self, embeddings):
        """
        Add embeddings to the index.
        embeddings: np.ndarray of shape (N, dimension) or (dimension,)
        """
        embeddings_clean = np.array(embeddings, dtype=np.float32)
        if len(embeddings_clean.shape) == 1:
            embeddings_clean = embeddings_clean.reshape(1, -1)
            
        if self.metric == 'cosine':
            # Normalize embeddings for cosine similarity
            norms = np.linalg.norm(embeddings_clean, axis=1, keepdims=True)
            norms[norms == 0] = 1e-8
            embeddings_clean = embeddings_clean / norms
            
        self.embeddings.append(embeddings_clean)
        
        if FAISS_AVAILABLE and self.index is not None:
            self.index.add(embeddings_clean)
            
    def finalize(self):
        """
        Concatenates all parts of embeddings for search scan fallback.
        """
        if self.embeddings:
            self.all_embeddings = np.vstack(self.embeddings)
        else:
            self.all_embeddings = np.empty((0, self.dimension), dtype=np.float32)
            
    def search(self, query_embedding, k=5):
        """
        Search the index for the top-k nearest neighbors.
        Returns:
            indices (np.ndarray): Indices of the top-k matches
            distances (np.ndarray): Similarity scores or distances of the top-k matches
        """
        query_np = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        if self.metric == 'cosine':
            q_norm = np.linalg.norm(query_np)
            if q_norm > 0:
                query_np = query_np / q_norm
                
        if FAISS_AVAILABLE and self.index is not None:
            distances, indices = self.index.search(query_np, k)
            return indices[0], distances[0]
        else:
            return self._numpy_search(query_np, k)
            
    def _numpy_search(self, query_np, k):
        if self.all_embeddings is None or self.all_embeddings.shape[0] == 0:
            self.finalize()
            
        if self.all_embeddings.shape[0] == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
            
        # Limit k to total number of embeddings
        k = min(k, self.all_embeddings.shape[0])
        if k <= 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
            
        if self.metric == 'cosine':
            similarities = np.dot(self.all_embeddings, query_np.T).squeeze(axis=1)
            top_k_indices = np.argsort(similarities)[::-1][:k]
            return top_k_indices, similarities[top_k_indices]
        else:
            diffs = self.all_embeddings - query_np
            dists = np.linalg.norm(diffs, axis=1)
            top_k_indices = np.argsort(dists)[:k]
            return top_k_indices, dists[top_k_indices]
            
    def benchmark_search(self, query_embedding, k=5, runs=100):
        """
        Compares search speed of NumPy brute force vs FAISS index.
        Returns a dict with latency statistics.
        """
        query_np = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        if self.metric == 'cosine':
            q_norm = np.linalg.norm(query_np)
            if q_norm > 0:
                query_np = query_np / q_norm
                
        if self.all_embeddings is None or self.all_embeddings.shape[0] == 0:
            self.finalize()
            
        if self.all_embeddings.shape[0] == 0:
            return {
                'numpy_latency_ms': 0.0,
                'faiss_latency_ms': 0.0,
                'faiss_available': FAISS_AVAILABLE
            }
            
        # NumPy benchmark
        t_start_np = time.perf_counter()
        for _ in range(runs):
            np_indices, np_dists = self._numpy_search(query_np, k)
        t_end_np = time.perf_counter()
        np_time = ((t_end_np - t_start_np) / runs) * 1000.0  # in ms
        
        # FAISS benchmark
        if FAISS_AVAILABLE and self.index is not None:
            t_start_faiss = time.perf_counter()
            for _ in range(runs):
                # Ensure the same parameters are passed to Index search
                distances, indices = self.index.search(query_np, k)
            t_end_faiss = time.perf_counter()
            faiss_time = ((t_end_faiss - t_start_faiss) / runs) * 1000.0  # in ms
        else:
            faiss_time = -1.0
            
        return {
            'numpy_latency_ms': np_time,
            'faiss_latency_ms': faiss_time,
            'faiss_available': FAISS_AVAILABLE and self.index is not None
        }
