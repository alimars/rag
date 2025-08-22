from langchain_ollama import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
import langid

class OfflineTranslationSystem:
    def __init__(self):
        print("Initializing OfflineTranslationSystem...")
        import os
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        print(f"Using Ollama base URL for translator: {ollama_base_url}")
        translator_model = os.getenv("TRANSLATOR_MODEL", "mistral-nemo:latest")
        print(f"Using translator model: {translator_model}")
        self.translator = OllamaLLM(model=translator_model, base_url=ollama_base_url)
        self.detector = langid
        self.supported_languages = {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "ru": "Russian",
            "ar": "Arabic"
        }
    
    def detect_language(self, text):
        lang, _ = self.detector.classify(text)
        return lang
    
    def translate(self, text, target_lang):
        prompt = f"""
        <|im_start|>system
        You are a professional translator. Rules:
        1. Translate exactly without adding/removing content
        2. Preserve technical terminology
        3. Maintain original formatting
        4. Output ONLY the translation
        Target language: {target_lang} ({self.supported_languages.get(target_lang, '')})
        <|im_end|>
        <|im_start|>user
        {text}
        <|im_end|>
        <|im_start|>assistant
        """
        return self.translator.invoke(prompt).strip()
    
    def translate_query(self, query, doc_language):
        q_lang = self.detect_language(query)
        if q_lang != doc_language:
            print(f"Translating query from {q_lang} to {doc_language}")
            return self.translate(query, doc_language)
        return query
    
def embed_text(texts):
    """Embed texts with caching to prevent unnecessary API calls"""
    import os
    import hashlib
    import pickle
    import concurrent.futures
    import threading
    import time
    
    print(f"📝 embed_text called with {len(texts)} texts")
    
    # Get cache directory
    CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Generate cache key based on texts and model
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embedding_model = os.getenv("EMBEDDING_MODEL", "jeffh/intfloat-multilingual-e5-large:Q8_0")
    
    # Create a hash of the texts and model parameters
    hash_input = f"{sorted(texts)}_{ollama_base_url}_{embedding_model}"
    cache_key = hashlib.md5(hash_input.encode()).hexdigest()
    cache_file = os.path.join(CACHE_DIR, f"embeddings_{cache_key}.pkl")
    
    # Try to load from cache first
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cached_result = pickle.load(f)
            print(f"✅ Loaded embeddings for {len(texts)} texts from cache")
            return cached_result
        except Exception as e:
            print(f"Error loading embeddings cache: {str(e)}")
            # Remove corrupted cache file
            os.remove(cache_file)
    
    print(f".Embedding {len(texts)} texts with {embedding_model} model")
    
    # Define embedding function with timeout
    def do_embedding():
        try:
            print(f"🔧 Initializing OllamaEmbeddings with model: {embedding_model} and base_url: {ollama_base_url}")
            embeddings = OllamaEmbeddings(
                model=embedding_model,
                base_url=ollama_base_url
            )
            print(f"🔧 OllamaEmbeddings initialized successfully")
            
            # Check if texts list is empty
            if not texts:
                print("⚠️  No texts to embed, returning empty list")
                return []
            
            print(f"🔧 Starting embedding of {len(texts)} texts")
            result = embeddings.embed_documents(texts)
            print(f"🔧 Embedding completed successfully for {len(texts)} texts")
            
            # Validate result
            if result is None:
                print("⚠️  Embedding result is None")
                raise ValueError("Embedding result is None")
            
            if not isinstance(result, list):
                print(f"⚠️  Embedding result is not a list: {type(result)}")
                raise ValueError(f"Embedding result is not a list: {type(result)}")
            
            print(f"🔧 Returning {len(result)} embeddings")
            return result
        except Exception as e:
            print(f"❌ Error creating embeddings: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            raise
    
    # Run embedding with timeout
    try:
        # Retry logic with exponential backoff
        max_retries = 3
        base_delay = 5  # seconds
        result = None
        
        for attempt in range(max_retries):
            try:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(do_embedding)
                    result = future.result(timeout=120 * (attempt + 1))  # Increase timeout with each attempt
                print(f"Successfully embedded {len(texts)} texts")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"⚠️ Embedding attempt {attempt + 1} failed: {str(e)}")
                print(f"🔄 Retrying in {base_delay * (2 ** attempt)} seconds...")
                time.sleep(base_delay * (2 ** attempt))
        
        # Save to cache if we have a result
        if result is not None:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
                print(f"💾 Saved embeddings to cache")
            except Exception as e:
                print(f"Warning: Could not save embeddings to cache: {str(e)}")
            
            return result
        else:
            print("❌ Failed to generate embeddings after all retries")
            return None
    except concurrent.futures.TimeoutError:
        print(f"❌ Embedding timed out after 2 minutes")
        raise Exception(f"Embedding operation timed out. This might be due to a large number of texts ({len(texts)}) or connectivity issues with Ollama.")
    except Exception as e:
        print(f"Error embedding texts: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise