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
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        reranker_model = os.getenv("RERANKER_MODEL", "mxbai-rerank-large")
        self.reranker = OllamaLLM(base_url=ollama_base_url, model=reranker_model, temperature=0)
    
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
        score_map = {}  # Map to store doc_id -> similarity score mappings
        for rank, docs in enumerate(rankings):
            for i, (doc, similarity_score) in enumerate(docs):
                doc_id = id(doc)
                doc_map[doc_id] = doc  # Store the actual document
                score_map[doc_id] = similarity_score  # Store the similarity score
                fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1/(k + i + rank + 1)
        # Return both scores and documents
        sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_map[doc_id], fused_score, score_map[doc_id]) for doc_id, fused_score in sorted_docs]
    
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
        # Extract documents and scores from fused results
        fused_docs_with_scores = [(doc, similarity_score) for doc, fused_score, similarity_score in fused[:top_k*2]]
        
        # Re-ranking
        rerank_prompt = "\n".join([
            f"Document {i+1}: {doc.page_content[:500]}{'...' if len(doc.page_content) > 500 else ''}"
            for i, (doc, score) in enumerate(fused_docs_with_scores)
        ])
        rerank_prompt = f"Query: {query}\n\nRank these by relevance (comma-separated numbers 1-{len(fused_docs_with_scores)}):\n{rerank_prompt}"
        
        try:
            ranking_order = self.reranker.invoke(rerank_prompt)
            ranked_ids = [int(x.strip()) - 1 for x in ranking_order.split(",")]
            reranked_results = [fused_docs_with_scores[i] for i in ranked_ids[:top_k]]
        except Exception as e:
            print(f"Re-ranking failed: {str(e)}")
            reranked_results = fused_docs_with_scores[:top_k]
        
        # Format results with metadata
        results = []
        for doc, similarity_score in reranked_results:
            # Extract metadata if available
            metadata = getattr(doc, 'metadata', {})
            
            # Create structured result
            result = {
                'content': doc.page_content,
                'score': similarity_score,
                'metadata': metadata
            }
            
            # Add specific metadata fields if available
            if 'page' in metadata:
                result['page'] = metadata['page']
            else:
                result['page'] = 1  # Default to page 1 if not available
            if 'chunk_id' in metadata:
                result['chunk_id'] = metadata['chunk_id']
            else:
                # Generate a default chunk_id if not available
                source = metadata.get('source', 'unknown')
                filename = os.path.basename(source)
                filename_without_ext = os.path.splitext(filename)[0]
                result['chunk_id'] = f"{filename_without_ext}_chunk_0000"
                
            results.append(result)
        
        # Sort results by score in descending order
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Save to cache
        self.save_to_cache(cache_key, results)
        print("💾 Saved retrieval results to cache")
        return results