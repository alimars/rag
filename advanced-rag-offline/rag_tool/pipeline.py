from rag_tool.document_processor import load_documents, chunk_text, raptor_clustering
from rag_tool.indexing import MultiRepresentationIndex
from rag_tool.retrieval import RetrievalSystem
from rag_tool.translation import OfflineTranslationSystem
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
import os
import hashlib
import pickle

# Cache directory
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Global pipeline instance
_PIPELINE_INSTANCE = None

class FocusedRAGPipeline:
    def __new__(cls, data_path, language="en"):
        """Ensure only one pipeline instance is created and shared"""
        global _PIPELINE_INSTANCE
        if _PIPELINE_INSTANCE is None:
            _PIPELINE_INSTANCE = super(FocusedRAGPipeline, cls).__new__(cls)
        return _PIPELINE_INSTANCE
    
    def __init__(self, data_path, language="en"):
        # Check if already initialized to prevent duplicate initialization
        if hasattr(self, '_initialized'):
            return
            
        self.data_path = data_path
        self.language = language
        self.index = None
        self.retriever = None
        print("Initializing generator with llama3:8b model...")
        import os
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        generator_model = os.getenv("GENERATOR_MODEL", "llama3:8b")
        print(f"Using Ollama base URL for generator: {ollama_base_url}")
        print(f"Using generator model: {generator_model}")
        self.generator = OllamaLLM(model=generator_model, base_url=ollama_base_url)
        self.translator = OfflineTranslationSystem()
        self.is_initialized = False
        self._initialized = True
    
    def get_cache_key(self, question, target_lang=None):
        """Generate a cache key based on question and parameters"""
        hash_input = f"{question}_{target_lang}_{self.language}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def save_to_cache(self, key, data):
        """Save query results to cache"""
        cache_file = os.path.join(CACHE_DIR, f"query_{key}.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        return cache_file
    
    def load_from_cache(self, key):
        """Load query results from cache"""
        cache_file = os.path.join(CACHE_DIR, f"query_{key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading query cache {key}: {str(e)}")
                # Remove corrupted cache file
                os.remove(cache_file)
        return None
    
    def initialize(self):
        if self.is_initialized:
            print("✅ Pipeline already initialized, skipping initialization")
            return True
            
        print("🔄 Initializing RAG pipeline...")
        try:
            print("Loading documents...")
            docs = load_documents(self.data_path, self.language)
            print(f"📄 Processed {len(docs)} documents")
            print(f"📝 Document sources: {[doc.metadata.get('source', 'Unknown') for doc in docs]}")
        except Exception as e:
            print(f"❌ Failed to load documents: {str(e)}")
            raise
        
        try:
            print("Chunking text...")
            chunks = chunk_text(docs)
            print(f"🧩 Created {len(chunks)} chunks")
        except Exception as e:
            print(f"❌ Failed to chunk text: {str(e)}")
            raise
        
        try:
            print("Building RAPTOR hierarchy...")
            raptor_chunks = raptor_clustering(chunks)
            print(f"🌲 Built {len(raptor_chunks)} hierarchical clusters")
        except Exception as e:
            print(f"❌ Failed to build RAPTOR clusters: {str(e)}")
            raise
        
        try:
            print("🏗️ Constructing indexes...")
            self.index = MultiRepresentationIndex()
            self.index.build_indexes(chunks, raptor_chunks)
        except Exception as e:
            print(f"❌ Failed to build indexes: {str(e)}")
            raise
        
        try:
            print("🔍 Initializing retriever...")
            self.retriever = RetrievalSystem(self.index)
        except Exception as e:
            print(f"❌ Failed to initialize retriever: {str(e)}")
            raise
        
        self.is_initialized = True
        print("✅ Pipeline initialized successfully")
        return True
    
    def query(self, question, target_lang=None, return_original=False):
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized")
            
        # Generate cache key
        cache_key = self.get_cache_key(question, target_lang)
        print(f"🔍 Checking pipeline cache for key: {cache_key}")
        
        # Try to load from cache first
        cached_data = self.load_from_cache(cache_key)
        if cached_data is not None:
            print("✅ Loaded query response from cache")
            return cached_data
        else:
            print("🔄 Cache miss - processing query")
            
        print(f"❓ Query: {question}")
        
        # If return_original is True, return the original documents directly
        if return_original:
            # Detect query language
            query_language = self.translator.detect_language(question)
            
            # Translate query if needed
            translated_query = question
            if query_language != self.language:
                print(f"Translating query from {query_language} to {self.language}")
                translated_query = self.translator.translate_query(question, self.language)
            print(f"🌐 Query language: {query_language}, Document language: {self.language}")
            print(f"🌐 Translated query: {translated_query}")
            
            # Retrieve relevant documents
            context_results = self.retriever.retrieve(translated_query)
            print(f"🔍 Retrieved {len(context_results)} documents")
            
            # Convert results to tuples for backward compatibility
            context_docs_with_scores = [
                (Document(page_content=result['content'], metadata=result['metadata']), result['score'])
                for result in context_results
            ]
            
            # Return original documents directly with metadata and scores
            original_content = "\n\n".join([
                f"Source: {doc.metadata.get('source', 'Unknown')}\n"
                f"Page: {doc.metadata.get('page', 1)}\n"
                f"Chunk ID: {doc.metadata.get('chunk_id', 'unknown_chunk_0000')}\n"
                f"Similarity Score: {score}\n"
                f"Content: {doc.page_content}"
                for doc, score in context_docs_with_scores
            ])
            
            # Prepare detailed results with metadata and scores
            detailed_results = []
            for doc, score in context_docs_with_scores:
                detailed_results.append({
                    "doc_name": doc.metadata.get('source', 'Unknown'),
                    "page": doc.metadata.get('page', 1),
                    "chunk_id": doc.metadata.get('chunk_id', 'unknown_chunk_0000'),
                    "similarity_score": float(score),
                    "content": doc.page_content
                })
            
            result = {
                "original_response": original_content,
                "detailed_results": detailed_results,
                "translation": None,
                "source_language": query_language
            }
            
            # Save to cache
            self.save_to_cache(cache_key, result)
            print("💾 Saved query response to cache")
            return result
        
        # Detect query language
        query_language = self.translator.detect_language(question)
        
        # Translate query if needed
        translated_query = question
        if query_language != self.language:
            print(f"Translating query from {query_language} to {self.language}")
            translated_query = self.translator.translate_query(question, self.language)
        print(f"🌐 Query language: {query_language}, Document language: {self.language}")
        print(f"🌐 Translated query: {translated_query}")
        
        # Retrieve relevant documents
        context_results = self.retriever.retrieve(translated_query)
        print(f"🔍 Retrieved {len(context_results)} documents")
        
        # Convert results to tuples for backward compatibility
        context_docs_with_scores = [
            (Document(page_content=result['content'], metadata=result['metadata']), result['score'])
            for result in context_results
        ]
        
        # Prepare context string with citations and scores
        context_str = "\n\n".join([
            f"📑 Source: {doc.metadata.get('source', 'Unknown')}\n"
            f"Similarity Score: {score}\n"
            f"Content: {doc.page_content}"
            for doc, score in context_docs_with_scores
        ])
        
        # Prepare detailed results with metadata and scores
        detailed_results = []
        for doc, score in context_docs_with_scores:
            detailed_results.append({
                "doc_name": doc.metadata.get('source', 'Unknown'),
                "page": doc.metadata.get('page', 1),
                "chunk_id": doc.metadata.get('chunk_id', 'unknown_chunk_0000'),
                "similarity_score": float(score),
                "content": doc.page_content
            })
        
        # Prepare language-specific instructions
        if self.language == "ar":
            language_instruction = "- يجب أن تكون إجابتك باللغة العربية (العربية فقط)"
        else:
            language_instruction = f"- Answer in the original document language ({self.language})"
        
        prompt = f"""
        **INSTRUCTIONS**
        - Answer concisely using ONLY the context below
        - Cite sources using [Source: filename] notation
        - If unsure, say "I couldn't find definitive information"
        {language_instruction}
        
        **QUESTION**
        {translated_query}
        
        **CONTEXT**
        {context_str}
        
        **ANSWER**
        """
        
        print("🤖 Generating response...")
        response = self.generator.invoke(prompt)
        
        # Add translation if requested
        translation = None
        if target_lang and target_lang != self.language:
            print(f"🌐 Translating response to {target_lang}...")
            translation = self.translator.translate(response, target_lang)
        
        result = {
            "original_response": response,
            "detailed_results": detailed_results,
            "translation": translation,
            "source_language": query_language
        }
        
        # Save to cache
        self.save_to_cache(cache_key, result)
        print("💾 Saved query response to cache")
        return result