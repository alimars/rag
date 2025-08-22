from rag_tool.query_transformer import QueryTransformer
from langchain_ollama import OllamaLLM
import numpy as np
import os
import hashlib
import pickle
from pathlib import Path

# Cache directory
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

class RetrievalSystem:
    def __init__(self, index):
        self.index = index
        print(f"🛠️ RetrievalSystem initialized with index: {type(index)}")
        # Check if RAPTOR is enabled
        if hasattr(index, 'raptor_index') and index.raptor_index is None:
            print("⚠️  RAPTOR retrieval is currently disabled")
    
    def get_cache_key(self, query, top_k=10):
        """Generate a cache key based on query and parameters"""
        hash_input = f"{query}_{top_k}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def save_to_cache(self, key, data):
        """Save retrieval results to cache"""
        cache_file = os.path.join(CACHE_DIR, f"retrieval_{key}.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        return cache_file
    
    def load_from_cache(self, key):
        """Load retrieval results from cache"""
        cache_file = os.path.join(CACHE_DIR, f"retrieval_{key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading retrieval cache {key}: {str(e)}")
                # Remove corrupted cache file
                os.remove(cache_file)
        return None
    
    def reciprocal_rank_fusion(self, rankings, k=60):
        fused_scores = {}
        doc_map = {}  # Map to store doc_id -> doc mappings
        for rank, docs in enumerate(rankings):
            for i, doc in enumerate(docs):
                doc_id = id(doc)
                doc_map[doc_id] = doc  # Store the actual document
                fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1/(k + i + rank + 1)
        # Return both scores and documents
        sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_id, score, doc_map[doc_id]) for doc_id, score in sorted_docs]
    
    def retrieve(self, query, top_k=10):
        # Generate cache key
        cache_key = self.get_cache_key(query, top_k)
        print(f"🔍 Checking retrieval cache for key: {cache_key}")
        
        # Try to load from cache first
        cached_data = self.load_from_cache(cache_key)
        if cached_data is not None:
            print("🔍 Loaded retrieval results from cache")
            return cached_data
        else:
            print("🔄 Retrieval cache miss - processing retrieval")
        
        print("🔍 Retrieving documents...")
        # Generate queries
        transformer = QueryTransformer()
        
        # Original query retrieval
        original_results = self.index.hybrid_search(query, top_k*3)
        
        # Multi-query retrieval
        multi_queries = transformer.multi_query(query)
        multi_results = [self.index.hybrid_search(q, top_k) for q in multi_queries]
        
        # Query decomposition
        sub_queries = transformer.decompose_query(query)
        decomposed_results = [self.index.hybrid_search(q, top_k) for q in sub_queries]
        
        # RAG-Fusion
        all_rankings = [original_results] + multi_results + decomposed_results
        fused = self.reciprocal_rank_fusion(all_rankings)
        fused_docs = [doc for doc_id, score, doc in fused[:top_k*2]]
        
        # No reranking, just take top results from fusion
        results = fused_docs[:top_k]
        
        # Save to cache
        self.save_to_cache(cache_key, results)
        print("💾 Saved retrieval results to cache")
        return results