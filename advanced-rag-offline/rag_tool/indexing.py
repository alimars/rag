from langchain_community.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
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
        self.raptor_index = None
        self.documents = []
        
    def close(self):
        """Properly close Chroma clients and clean up resources"""
        try:
            if self.dense_index:
                self.dense_index.delete_collection()
        except Exception as e:
            print(f"Error cleaning dense index: {str(e)}")
        finally:
            self.dense_index = None

        try:
            if self.raptor_index:
                self.raptor_index.delete_collection()
        except Exception as e:
            print(f"Error cleaning RAPTOR index: {str(e)}")
        finally:
            self.raptor_index = None
        
    def get_cache_key(self, chunks, raptor_chunks):
        """Generate a cache key based on chunk content"""
        # Create hash of chunk content
        chunk_content_hash = hashlib.md5(str([c.page_content for c in chunks]).encode()).hexdigest()
        raptor_content_hash = hashlib.md5(str([c.page_content for c in raptor_chunks]).encode()).hexdigest()
        
        # Get embedding model from environment
        # Sanitize model name for Windows paths

        embedding_model = os.getenv("EMBEDDING_MODEL", "jeffh/intfloat-multilingual-e5-large:q8_0")
        
        # Sanitize embedding model name for file system compatibility
        sanitized_model = embedding_model.replace("/", "-").replace(":", "_")
        
        return f"{chunk_content_hash}_{raptor_content_hash}_{sanitized_model}"
        
    def save_to_cache(self, key, data):
        """Save index data to cache"""
        cache_file = os.path.join(CACHE_DIR, f"index_{key}.pkl")
        # Save only the serializable parts
        cache_data = {
            'documents': self.documents
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
            self.documents = cached_data['documents']
# self.sparse_index = cached_data['sparse_index']
            # We still need to rebuild Chroma indexes as they're not easily serializable
            # But we can check if the Chroma DB already exists to avoid rebuilding
            self._rebuild_chroma_indexes(chunks, raptor_chunks)
            return
        
        print("🏗️ Constructing indexes...")
        self.documents = [c.page_content for c in chunks]
        
        # Get Ollama base URL from environment
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        # Sanitize model name for Windows paths
        # Sanitize model name for Windows paths
        # Sanitize model name for Windows paths
        embedding_model = os.getenv("EMBEDDING_MODEL", "jeffh/intfloat-multilingual-e5-large:q8_0")
        
        import concurrent.futures
        
        # Dense embeddings index with Chroma instead of FAISS
        try:
            print("Creating dense index...")
            # Check if Chroma DB already exists to avoid rebuilding
            # Sanitize directory name for Windows
            sanitized_key = cache_key.replace(":", "_").replace("/", "-")[:50]
            chroma_persist_dir = os.path.join(CACHE_DIR, f"dense_{sanitized_key}")
            if os.path.exists(chroma_persist_dir):
                print("Loading existing dense index from disk...")
                dense_embeddings = OllamaEmbeddings(model=embedding_model, base_url=ollama_base_url)
                self.dense_index = Chroma(
                    persist_directory=chroma_persist_dir,
                    embedding_function=dense_embeddings,
                    collection_name="dense_index"
                )
            else:
                def create_dense_index():
                    dense_embeddings = OllamaEmbeddings(model=embedding_model, base_url=ollama_base_url)
                    chroma_index = Chroma.from_documents(
                        chunks,
                        dense_embeddings,
                        collection_name="dense_index",
                        persist_directory=chroma_persist_dir
                    )
                    return chroma_index
                
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
            tokenized_docs = [doc.split() for doc in self.documents]
            print("✅ Sparse index created successfully")
        except Exception as e:
            print(f"❌ Failed to create sparse index: {str(e)}")
            raise
        
        # RAPTOR index (hierarchical) - DISABLED
        try:
            print("⚠️  RAPTOR index creation is currently disabled")
            self.raptor_index = None
            # Original implementation (commented out for later use):
            # print("Creating RAPTOR index...")
            # # Check if Chroma DB already exists to avoid rebuilding
            # # Sanitize directory name for Windows
            # sanitized_key = cache_key.replace(":", "_").replace("/", "-")[:50]
            # chroma_raptor_dir = os.path.join(CACHE_DIR, f"raptor_{sanitized_key}")
            # if os.path.exists(chroma_raptor_dir):
            #     print("Loading existing RAPTOR index from disk...")
            #     raptor_embeddings = OllamaEmbeddings(model=embedding_model, base_url=ollama_base_url)
            #     self.raptor_index = Chroma(
            #         persist_directory=chroma_raptor_dir,
            #         embedding_function=raptor_embeddings,
            #         collection_name="raptor_index"
            #     )
            # else:
            #     def create_raptor_index():
            #         print(f"🔧 Creating RAPTOR index with model: {embedding_model} and base_url: {ollama_base_url}")
            #         raptor_embeddings = OllamaEmbeddings(model=embedding_model, base_url=ollama_base_url)
            #         chroma_index = Chroma.from_documents(
            #             raptor_chunks,
            #             raptor_embeddings,
            #             collection_name="raptor_index",
            #             persist_directory=chroma_raptor_dir
            #         )
            #         return chroma_index
            #
            #     with concurrent.futures.ThreadPoolExecutor() as executor:
            #         future = executor.submit(create_raptor_index)
            #         self.raptor_index = future.result(timeout=120)  # 2 minute timeout
            #         print(f"RAPTOR index metadata: {self.raptor_index._collection.metadata}")  # Debug Chroma config
            #         print(f"First 3 RAPTOR chunks: {raptor_chunks[:3]}")  # Verify chunk structure
            # print("✅ RAPTOR index created successfully")
        except Exception as e:
            print(f"⚠️  RAPTOR index creation skipped (disabled): {str(e)}")
            self.raptor_index = None
        
        # Save to cache (only serializable parts)
        try:
            cache_data = {
                'documents': self.documents,
# 'sparse_index': self.sparse_index,
            }
            self.save_to_cache(cache_key, cache_data)
            print("💾 Saved indexes to cache")
        except Exception as e:
            print(f"Warning: Could not save indexes to cache: {str(e)}")
        
    def _rebuild_chroma_indexes(self, chunks, raptor_chunks):
        """Rebuild Chroma indexes from cached data"""
        print("🔄 Rebuilding Chroma indexes...")
        # Get Ollama base URL from environment
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        embedding_model = os.getenv("EMBEDDING_MODEL", "jeffh/intfloat-multilingual-e5-large:Q8_0")
        
        # Generate cache key for Chroma directories
        cache_key = self.get_cache_key(chunks, raptor_chunks)
        
        # Check if Chroma DB already exists to avoid rebuilding
        chroma_persist_dir = os.path.join(CACHE_DIR, f"chroma_dense_{cache_key}")
        if os.path.exists(chroma_persist_dir):
            print("Loading existing dense index from disk...")
            dense_embeddings = OllamaEmbeddings(model=embedding_model, base_url=ollama_base_url)
            self.dense_index = Chroma(
                persist_directory=chroma_persist_dir,
                embedding_function=dense_embeddings,
                collection_name="dense_index"
            )
        else:
            # Dense embeddings index with Chroma instead of FAISS
            dense_embeddings = OllamaEmbeddings(model=embedding_model, base_url=ollama_base_url)
            self.dense_index = Chroma.from_documents(
                chunks,
                dense_embeddings,
                collection_name="dense_index",
                persist_directory=chroma_persist_dir
            )
        
        # Check if RAPTOR Chroma DB already exists to avoid rebuilding
        chroma_raptor_dir = os.path.join(CACHE_DIR, f"chroma_raptor_{cache_key}")
        if os.path.exists(chroma_raptor_dir):
            print("Loading existing RAPTOR index from disk...")
            raptor_embeddings = OllamaEmbeddings(model=embedding_model, base_url=ollama_base_url)
            self.raptor_index = Chroma(
                persist_directory=chroma_raptor_dir,
                embedding_function=raptor_embeddings,
                collection_name="raptor_index"
            )
        else:
            # RAPTOR index (hierarchical)
            raptor_embeddings = OllamaEmbeddings(model=embedding_model, base_url=ollama_base_url)
            self.raptor_index = Chroma.from_documents(
                raptor_chunks,
                raptor_embeddings,
                collection_name="raptor_index",
                persist_directory=chroma_raptor_dir
            )

    def hybrid_search(self, query, top_k=10):
        if not self.dense_index:
            raise ValueError("dense_index not initialized in hybrid_search()")
        # RAPTOR index can be None if disabled
        # if not self.raptor_index:
        #     raise ValueError("raptor_index not initialized in hybrid_search()")
            
        print(f"🔎 Hybrid search - dense_index: {type(self.dense_index)}, raptor_index: {type(self.raptor_index)}")
        # Dense retrieval
        dense_results = self.dense_index.similarity_search(query, k=top_k*2)
        
        # Sparse retrieval
        tokenized_query = query.split()
        
        # RAPTOR retrieval (skip if disabled)
        if self.raptor_index is not None:
            raptor_results = self.raptor_index.similarity_search(query, k=top_k)
        else:
            print("⚠️  RAPTOR retrieval skipped (disabled)")
            raptor_results = []
        
        # Combine results
        all_results = list(dense_results) + raptor_results
        return all_results[:top_k*3]
