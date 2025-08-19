from langchain_community.vectorstores import Chroma
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
        self.documents = []
        
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
            self.documents = cached_data['documents']
            self.sparse_index = cached_data['sparse_index']
            # We still need to rebuild Chroma indexes as they're not easily serializable
            # But we can check if the Chroma DB already exists to avoid rebuilding
            self._rebuild_chroma_indexes(chunks, raptor_chunks)
            return
        
        print("🏗️ Constructing indexes...")
        self.documents = [c.page_content for c in chunks]
        
        # Get Ollama base URL from environment
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        
        import concurrent.futures
        
        # Dense embeddings index with Chroma instead of FAISS
        try:
            print("Creating dense index...")
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
                def create_dense_index():
                    dense_embeddings = OllamaEmbeddings(model=embedding_model, base_url=ollama_base_url)
                    chroma_index = Chroma.from_documents(
                        chunks,
                        dense_embeddings,
                        collection_name="dense_index",
                        persist_directory=chroma_persist_dir
                    )
                    chroma_index.persist()
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
            self.sparse_index = BM25Okapi(tokenized_docs)
            print("✅ Sparse index created successfully")
        except Exception as e:
            print(f"❌ Failed to create sparse index: {str(e)}")
            raise
        
        # RAPTOR index (hierarchical)
        try:
            print("Creating RAPTOR index...")
            # Check if Chroma DB already exists to avoid rebuilding
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
                def create_raptor_index():
                    chroma_index = Chroma.from_documents(
                        raptor_chunks,
                        OllamaEmbeddings(model=embedding_model, base_url=ollama_base_url),
                        collection_name="raptor_index",
                        persist_directory=chroma_raptor_dir
                    )
                    chroma_index.persist()
                    return chroma_index
                
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
                'documents': self.documents,
                'sparse_index': self.sparse_index,
            }
            self.save_to_cache(cache_key, cache_data)
            print("💾 Saved indexes to cache")
        except Exception as e:
            print(f"Warning: Could not save indexes to cache: {str(e)}")
        
    def _rebuild_chroma_indexes(self, chunks, raptor_chunks):
        """Rebuild Chroma indexes from cached data"""
        print("🔄 Rebuilding Chroma indexes...")
        # Get Ollama base URL from environment
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        
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
            self.dense_index.persist()
        
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
            self.raptor_index = Chroma.from_documents(
                raptor_chunks,
                OllamaEmbeddings(model=embedding_model, base_url=ollama_base_url),
                collection_name="raptor_index",
                persist_directory=chroma_raptor_dir
            )
            self.raptor_index.persist()

    def hybrid_search(self, query, top_k=10):
        # Dense retrieval
        dense_results = self.dense_index.similarity_search(query, k=top_k*2)
        
        # Sparse retrieval
        tokenized_query = query.split()
        sparse_scores = self.sparse_index.get_scores(tokenized_query)
        sparse_indices = np.argsort(sparse_scores)[::-1][:top_k*2]
        sparse_results = [self.documents[i] for i in sparse_indices]
        
        # RAPTOR retrieval
        raptor_results = self.raptor_index.similarity_search(query, k=top_k)
        
        # Combine results
        all_results = list(dense_results) + raptor_results
        return all_results[:top_k*3]