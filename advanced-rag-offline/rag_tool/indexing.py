from langchain_community.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings
from rank_bm25 import BM25Okapi
import numpy as np
import os
import pickle
import hashlib

# Cache directory
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

class MultiRepresentationIndex:
    def __init__(self):
        self.dense_index = None
        self.sparse_index = None
        self.raptor_index = None
        self.documents = []  # Store Document objects, not strings
        self.document_contents = []  # Store string contents for BM25
        
    def get_cache_key(self, chunks, raptor_chunks):
        """Generate a cache key based on chunk content"""
        # Create hash of chunk content
        chunk_content_hash = hashlib.md5(str([c.page_content for c in chunks]).encode()).hexdigest()
        raptor_content_hash = hashlib.md5(str([c.page_content for c in raptor_chunks]).encode()).hexdigest()
        
        # Get embedding model from environment
        embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        
        return f"{chunk_content_hash}_{raptor_content_hash}_{embedding_model}"
        
    def save_to_cache(self, key, data):
        """Save index data to cache"""
        cache_file = os.path.join(CACHE_DIR, f"index_{key}.pkl")
        # Save only the serializable parts
        cache_data = {
            'documents': self.documents,
            'sparse_index': data.get('sparse_index', None)
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        return cache_file
        
    def load_from_cache(self, key):
        """Load index data from cache"""
        cache_file = os.path.join(CACHE_DIR, f"index_{key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading index cache {key}: {str(e)}")
                # Remove corrupted cache file
                os.remove(cache_file)
        return None

    def build_indexes(self, chunks, raptor_chunks):
        # Generate cache key
        cache_key = self.get_cache_key(chunks, raptor_chunks)
        
        # Try to load from cache first
        cached_data = self.load_from_cache(cache_key)
        if cached_data is not None:
            print("🏗️ Loaded indexes from cache")
            self.document_contents = cached_data['documents']  # Load string contents
            self.documents = chunks  # Use current Document objects
            self.sparse_index = cached_data['sparse_index']
            # We still need to rebuild FAISS indexes as they're not easily serializable
            # But we can check if the FAISS index already exists to avoid rebuilding
            self._rebuild_faiss_indexes(chunks, raptor_chunks)
            return
        
        # Only print this message when actually constructing indexes, not when loading from cache
        print("🏗️ Constructing indexes...")
        self.documents = chunks  # Store Document objects
        self.document_contents = [c.page_content for c in chunks]  # Store string contents for BM25
        
        # Get Ollama base URL from environment
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        
        import concurrent.futures
        
        # Dense embeddings index with FAISS
        try:
            print("Creating dense index...")
            # Check if FAISS index already exists to avoid rebuilding
            faiss_index_file = os.path.join(CACHE_DIR, f"faiss_dense_{cache_key}")
            if os.path.exists(faiss_index_file):
                print("Loading existing dense index from disk...")
                dense_embeddings = OllamaEmbeddings(model=embedding_model, base_url=ollama_base_url)
                self.dense_index = FAISS.load_local(faiss_index_file, dense_embeddings, allow_dangerous_deserialization=True)
            else:
                def create_dense_index():
                    dense_embeddings = OllamaEmbeddings(model=embedding_model, base_url=ollama_base_url)
                    faiss_index = FAISS.from_documents(chunks, dense_embeddings)
                    faiss_index.save_local(faiss_index_file)
                    return faiss_index
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(create_dense_index)
                    self.dense_index = future.result(timeout=120)  # 2 minute timeout
            print("✅ Dense index created successfully")
        except concurrent.futures.TimeoutError:
            print("❌ Dense index creation timed out")
            raise Exception("Dense index creation timed out. This might be due to a large number of chunks or connectivity issues with Ollama.")
        except Exception as e:
            print(f"❌ Failed to create dense index: {str(e)}")
            raise
        
        # Sparse BM25 index
        try:
            print("Creating sparse index...")
            tokenized_docs = [doc.split() for doc in self.document_contents]
            self.sparse_index = BM25Okapi(tokenized_docs)
            print("✅ Sparse index created successfully")
        except Exception as e:
            print(f"❌ Failed to create sparse index: {str(e)}")
            raise
        
        # RAPTOR index (hierarchical)
        try:
            print("Creating RAPTOR index...")
            # Check if FAISS index already exists to avoid rebuilding
            faiss_raptor_file = os.path.join(CACHE_DIR, f"faiss_raptor_{cache_key}")
            if os.path.exists(faiss_raptor_file):
                print("Loading existing RAPTOR index from disk...")
                raptor_embeddings = OllamaEmbeddings(model=embedding_model, base_url=ollama_base_url)
                self.raptor_index = FAISS.load_local(faiss_raptor_file, raptor_embeddings, allow_dangerous_deserialization=True)
            else:
                def create_raptor_index():
                    faiss_index = FAISS.from_documents(
                        raptor_chunks,
                        OllamaEmbeddings(model=embedding_model, base_url=ollama_base_url)
                    )
                    faiss_index.save_local(faiss_raptor_file)
                    return faiss_index
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(create_raptor_index)
                    self.raptor_index = future.result(timeout=120)  # 2 minute timeout
            print("✅ RAPTOR index created successfully")
        except concurrent.futures.TimeoutError:
            print("❌ RAPTOR index creation timed out")
            raise Exception("RAPTOR index creation timed out. This might be due to a large number of chunks or connectivity issues with Ollama.")
        except Exception as e:
            print(f"❌ Failed to create RAPTOR index: {str(e)}")
            raise
        
        # Save to cache (only serializable parts)
        try:
            cache_data = {
                'documents': self.document_contents,  # Cache string contents, not Document objects
                'sparse_index': self.sparse_index,
            }
            self.save_to_cache(cache_key, cache_data)
            print("💾 Saved indexes to cache")
        except Exception as e:
            print(f"Warning: Could not save indexes to cache: {str(e)}")
        
    def _rebuild_faiss_indexes(self, chunks, raptor_chunks):
        """Rebuild FAISS indexes from cached data"""
        # Check if indexes are already loaded
        if self.dense_index is not None and self.raptor_index is not None:
            print("⏭️ FAISS indexes already loaded, skipping rebuild")
            return
        
        print("🔄 Rebuilding FAISS indexes...")
        # Get Ollama base URL from environment
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        
        # Generate cache key for FAISS directories
        cache_key = self.get_cache_key(chunks, raptor_chunks)
        
        # Check if FAISS index already exists to avoid rebuilding
        faiss_index_file = os.path.join(CACHE_DIR, f"faiss_dense_{cache_key}")
        if self.dense_index is None and os.path.exists(faiss_index_file):
            print("Loading existing dense index from disk...")
            dense_embeddings = OllamaEmbeddings(model=embedding_model, base_url=ollama_base_url)
            self.dense_index = FAISS.load_local(faiss_index_file, dense_embeddings, allow_dangerous_deserialization=True)
        elif self.dense_index is None:
            # Dense embeddings index with FAISS
            dense_embeddings = OllamaEmbeddings(model=embedding_model, base_url=ollama_base_url)
            self.dense_index = FAISS.from_documents(chunks, dense_embeddings)
            self.dense_index.save_local(faiss_index_file)
        
        # Check if RAPTOR FAISS index already exists to avoid rebuilding
        faiss_raptor_file = os.path.join(CACHE_DIR, f"faiss_raptor_{cache_key}")
        if self.raptor_index is None and os.path.exists(faiss_raptor_file):
            print("Loading existing RAPTOR index from disk...")
            raptor_embeddings = OllamaEmbeddings(model=embedding_model, base_url=ollama_base_url)
            self.raptor_index = FAISS.load_local(faiss_raptor_file, raptor_embeddings, allow_dangerous_deserialization=True)
        elif self.raptor_index is None:
            # RAPTOR index (hierarchical)
            self.raptor_index = FAISS.from_documents(
                raptor_chunks,
                OllamaEmbeddings(model=embedding_model, base_url=ollama_base_url)
            )
            self.raptor_index.save_local(faiss_raptor_file)

    def hybrid_search(self, query, top_k=10):
        """
        Perform hybrid search using dense, sparse, and RAPTOR indexes.
        
        Args:
            query (str): The search query
            top_k (int): Number of top results to return
            
        Returns:
            list: List of tuples (document, similarity_score) sorted by relevance
            
        Raises:
            Exception: If any search operation fails
        """
        try:
            # Dense retrieval with scores
            dense_results_with_scores = self.dense_index.similarity_search_with_score(query, k=top_k*2)
            dense_results = [(doc, score) for doc, score in dense_results_with_scores]
        except Exception as e:
            print(f"Error in dense retrieval: {str(e)}")
            dense_results = []
        
        try:
            # Sparse retrieval with scores
            tokenized_query = query.split()
            sparse_scores = self.sparse_index.get_scores(tokenized_query)
            sparse_indices = np.argsort(sparse_scores)[::-1][:top_k*2]
            # Create list of (document, score) tuples for sparse results
            sparse_results = [(self.documents[i], sparse_scores[i]) for i in sparse_indices]
        except Exception as e:
            print(f"Error in sparse retrieval: {str(e)}")
            sparse_results = []
        
        try:
            # RAPTOR retrieval with scores
            raptor_results_with_scores = self.raptor_index.similarity_search_with_score(query, k=top_k)
            raptor_results = [(doc, score) for doc, score in raptor_results_with_scores]
        except Exception as e:
            print(f"Error in RAPTOR retrieval: {str(e)}")
            raptor_results = []
        
        # Combine results
        all_results = dense_results + sparse_results + raptor_results
        
        # Sort by score (lower scores are better in FAISS)
        all_results.sort(key=lambda x: x[1])
        
        # Return top_k results
        return all_results[:top_k]