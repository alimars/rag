import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytesseract
from pdf2image import convert_from_path
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from rag_tool.translation import embed_text
import os
import pickle
import hashlib
import time
from typing import List, Tuple
import langid
from langdetect import detect, detect_langs
import re

# Map language codes to Tesseract codes
LANGUAGE_MAP = {
    "en": "eng",
    "es": "spa",
    "fr": "fra",
    "de": "deu",
    "ja": "jpn",
    "ko": "kor",
    "zh": "chi_sim",
    "ar": "ara",
    "ru": "rus"
}

def detect_document_language(text):
    """Detect document language using hybrid approach with langid and langdetect"""
    # Handle edge cases
    if not text or not text.strip():
        return "en"  # Default to English for empty text
    
    # For very short text, it's hard to detect language accurately
    if len(text.strip()) < 10:
        return "en"  # Default to English for very short text
    
    # Check if text contains Arabic characters using Unicode ranges
    def contains_arabic(text):
        arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
        return bool(arabic_pattern.search(text))
    
    # Check if text is primarily Arabic
    def is_arabic_dominant(text):
        arabic_chars = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', text))
        total_chars = len(re.findall(r'\S', text))  # Non-whitespace characters
        return arabic_chars > total_chars * 0.3 if total_chars > 0 else False
    
    try:
        # First check if text contains Arabic characters
        if contains_arabic(text):
            # If text is primarily Arabic, favor Arabic detection
            if is_arabic_dominant(text):
                return "ar"
        
        # Use both langid and langdetect for better accuracy
        langid_result = langid.classify(text)
        langid_lang, langid_confidence = langid_result
        
        # Get langdetect results with probabilities
        try:
            langdetect_results = detect_langs(text)
            langdetect_lang = langdetect_results[0].lang
            langdetect_confidence = langdetect_results[0].prob
        except:
            # If langdetect fails, use langid result
            langdetect_lang = langid_lang
            langdetect_confidence = 0.0
        
        # Confidence-based decision making
        # If one method is very confident, use it
        if langid_confidence > 0.9:
            selected_lang = langid_lang
        elif langdetect_confidence > 0.9:
            selected_lang = langdetect_lang
        # If both methods agree, use the result
        elif langid_lang == langdetect_lang:
            selected_lang = langid_lang
        # If one method detected Arabic and text contains Arabic characters, favor Arabic
        elif (langid_lang == "ar" or langdetect_lang == "ar") and contains_arabic(text):
            selected_lang = "ar"
        # If confidence levels are close, use the one with higher confidence
        elif langid_confidence >= langdetect_confidence:
            selected_lang = langid_lang
        else:
            selected_lang = langdetect_lang
        
        # Check if the detected language is in our supported languages
        if selected_lang in LANGUAGE_MAP:
            return selected_lang
        else:
            # Default to English if language is not supported
            return "en"
    except Exception as e:
        # If language detection fails, default to English
        print(f"Language detection failed: {str(e)}. Defaulting to English.")
        return "en"

# Cache directory
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_key(path: str, language: str = "en") -> str:
    """Generate a cache key based on path and language"""
    # Get all file paths and their modification times
    file_info = []
    for file_path in Path(path).rglob('*'):
        if file_path.is_file():
            file_info.append((str(file_path), file_path.stat().st_mtime))
    
    # Sort by file path for consistent hashing
    file_info.sort(key=lambda x: x[0])
    
    # Create hash of all file paths and modification times
    hash_input = f"{path}_{language}_" + "_".join([f"{fp}:{int(mt)}" for fp, mt in file_info])
    return hashlib.md5(hash_input.encode()).hexdigest()

def save_to_cache(key: str, data) -> str:
    """Save data to cache file"""
    cache_file = os.path.join(CACHE_DIR, f"{key}.pkl")
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    return cache_file

def load_from_cache(key: str):
    """Load data from cache file"""
    cache_file = os.path.join(CACHE_DIR, f"{key}.pkl")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading cache {key}: {str(e)}")
            # Remove corrupted cache file
            os.remove(cache_file)
    return None

def is_cache_valid(key: str) -> bool:
    """Check if cache exists and is valid"""
    cache_file = os.path.join(CACHE_DIR, f"{key}.pkl")
    return os.path.exists(cache_file)

def ocr_pdf(file_path, language):
    """Extract text from PDF using OCR"""
    tesseract_lang = LANGUAGE_MAP.get(language, "eng")
    images = convert_from_path(file_path)
    full_text = ""
    for i, img in enumerate(images):
        text = pytesseract.image_to_string(img, lang=tesseract_lang)
        full_text += f"Page {i+1}:\n{text}\n\n"
    return full_text

def load_documents(path, language="ar"):
    """Load and process documents from directory with caching"""
    # Generate cache key
    cache_key = get_cache_key(path, language)
    
    # Try to load from cache first
    cached_data = load_from_cache(f"documents_{cache_key}")
    if cached_data is not None:
        print("📄 Loaded documents from cache")
        return cached_data
    
    print("🔄 Loading and processing documents...")
    tesseract_lang = LANGUAGE_MAP.get(language, "ara")
    loaders = {
        '.pdf': UnstructuredLoader,
        '.docx': UnstructuredLoader,
        '.doc': UnstructuredLoader
    }
    documents = []
    
    import concurrent.futures
    import time
    
    # Process files with timeout
    file_paths = list(Path(path).rglob('*'))
    total_files = len([fp for fp in file_paths if fp.suffix.lower() in loaders])
    processed_files = 0
    
    for file_path in file_paths:
        if file_path.suffix.lower() in loaders:
            processed_files += 1
            print(f"Processing file {processed_files}/{total_files}: {file_path.name}")            
            def process_file(fp):
                try:
                    if fp.suffix.lower() == '.pdf':
                        try:
                            loader = UnstructuredLoader(str(fp))
                            docs = loader.load()
                            if docs and docs[0].page_content.strip():
                                # Combine all documents from the same file into a single document
                                combined_content = "\n\n".join([doc.page_content for doc in docs])
                                combined_metadata = docs[0].metadata.copy()
                                combined_metadata["source"] = str(fp)
                                
                                # Detect language for the document
                                detected_language = detect_document_language(combined_content)
                                combined_metadata["language"] = detected_language
                                
                                print(f"Detected language for {fp.name}: {detected_language}")
                                return [Document(page_content=combined_content, metadata=combined_metadata)]
                        except Exception as e:
                            print(f"UnstructuredLoader failed for {fp}: {str(e)}")
                            pass
                        # If UnstructuredLoader fails, use OCR with detected language
                        # For OCR, we'll use a default language for now, as we don't have text to detect language from
                        # In a more advanced implementation, we could do OCR first, then detect language
                        text = ocr_pdf(str(fp), "ara")  # Default to Arabic for OCR
                        metadata = {"source": str(fp)}
                        
                        # Detect language for the document
                        detected_language = detect_document_language(text)
                        metadata["language"] = detected_language
                        
                        print(f"Detected language for {fp.name} using OCR: {detected_language}")
                        return [Document(page_content=text, metadata=metadata)]
                    else:
                        loader = UnstructuredLoader(str(fp))
                        docs = loader.load()
                        if docs:
                            # Combine all documents from the same file into a single document
                            combined_content = "\n\n".join([doc.page_content for doc in docs])
                            combined_metadata = docs[0].metadata.copy()
                            combined_metadata["source"] = str(fp)
                            
                            # Detect language for the document
                            detected_language = detect_document_language(combined_content)
                            combined_metadata["language"] = detected_language
                            
                            print(f"Detected language for {fp.name}: {detected_language}")
                            return [Document(page_content=combined_content, metadata=combined_metadata)]
                        return docs
                except Exception as e:
                    print(f"Error processing {fp}: {str(e)}")
                    # Return empty list to continue with other files
                    return []
            
            try:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(process_file, file_path)
                    file_docs = future.result(timeout=300)  # 5 minute timeout per file
                    documents.extend(file_docs)
                    print(f"✅ Processed {file_path.name} successfully")
            except concurrent.futures.TimeoutError:
                print(f"❌ Processing {file_path.name} timed out after 5 minutes")
                print(f"⚠️  This file might be too large or corrupted. Consider splitting it into smaller parts.")
                # Continue with other files instead of failing completely
                continue
            except Exception as e:
                print(f"❌ Failed to process {file_path.name}: {str(e)}")
                # Continue with other files instead of failing completely
                continue
    
    # Save to cache
    try:
        save_to_cache(f"documents_{cache_key}", documents)
        print("💾 Saved documents to cache")
    except Exception as e:
        print(f"Warning: Could not save documents to cache: {str(e)}")
    
    # Filter complex metadata to avoid issues with Chroma
    filtered_documents = filter_complex_metadata(documents)
    
    return filtered_documents
    
    # Try to load from cache first
    cached_data = load_from_cache(f"documents_{cache_key}")
    if cached_data is not None:
        print("📄 Loaded documents from cache")
        return cached_data
    
    print("🔄 Loading and processing documents...")
    tesseract_lang = LANGUAGE_MAP.get(language, "eng")
    loaders = {
        '.pdf': UnstructuredLoader,
        '.docx': UnstructuredLoader,
        '.doc': UnstructuredLoader
    }
    documents = []
    
    import concurrent.futures
    import time
    
    # Process files with timeout
    file_paths = list(Path(path).rglob('*'))
    total_files = len([fp for fp in file_paths if fp.suffix.lower() in loaders])
    processed_files = 0
    
    for file_path in file_paths:
        if file_path.suffix.lower() in loaders:
            processed_files += 1
            print(f"Processing file {processed_files}/{total_files}: {file_path.name}")            
            def process_file(fp):
                try:
                    if fp.suffix.lower() == '.pdf':
                        try:
                            loader = UnstructuredLoader(str(fp))
                            docs = loader.load()
                            if docs and docs[0].page_content.strip():
                                # Combine all documents from the same file into a single document
                                combined_content = "\n\n".join([doc.page_content for doc in docs])
                                combined_metadata = docs[0].metadata.copy()
                                combined_metadata["source"] = str(fp)
                                
                                # Detect language for the document
                                detected_language = detect_document_language(combined_content)
                                combined_metadata["language"] = detected_language
                                
                                return [Document(page_content=combined_content, metadata=combined_metadata)]
                        except Exception as e:
                            print(f"UnstructuredLoader failed for {fp}: {str(e)}")
                            pass
                        # If UnstructuredLoader fails, use OCR with detected language
                        # For OCR, we'll use a default language for now, as we don't have text to detect language from
                        # In a more advanced implementation, we could do OCR first, then detect language
                        text = ocr_pdf(str(fp), "ara")  # Default to Arabic for OCR
                        metadata = {"source": str(fp)}
                        
                        # Detect language for the document
                        detected_language = detect_document_language(text)
                        metadata["language"] = detected_language
                        
                        return [Document(page_content=text, metadata=metadata)]
                    else:
                        loader = UnstructuredLoader(str(fp))
                        docs = loader.load()
                        if docs:
                            # Combine all documents from the same file into a single document
                            combined_content = "\n\n".join([doc.page_content for doc in docs])
                            combined_metadata = docs[0].metadata.copy()
                            combined_metadata["source"] = str(fp)
                            
                            # Detect language for the document
                            detected_language = detect_document_language(combined_content)
                            combined_metadata["language"] = detected_language
                            
                            return [Document(page_content=combined_content, metadata=combined_metadata)]
                        return docs
                except Exception as e:
                    print(f"Error processing {fp}: {str(e)}")
                    # Return empty list to continue with other files
                    return []
            
            try:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(process_file, file_path)
                    file_docs = future.result(timeout=300)  # 5 minute timeout per file
                    documents.extend(file_docs)
                    print(f"✅ Processed {file_path.name} successfully")
            except concurrent.futures.TimeoutError:
                print(f"❌ Processing {file_path.name} timed out after 5 minutes")
                print(f"⚠️  This file might be too large or corrupted. Consider splitting it into smaller parts.")
                # Continue with other files instead of failing completely
                continue
            except Exception as e:
                print(f"❌ Failed to process {file_path.name}: {str(e)}")
                # Continue with other files instead of failing completely
                continue
    
    # Save to cache
    try:
        save_to_cache(f"documents_{cache_key}", documents)
        print("💾 Saved documents to cache")
    except Exception as e:
        print(f"Warning: Could not save documents to cache: {str(e)}")
    
    # Filter complex metadata to avoid issues with Chroma
    filtered_documents = filter_complex_metadata(documents)
    
    return filtered_documents

def chunk_text(docs, chunk_size=1024, overlap=128):
    """Split documents into chunks with caching"""
    # Generate cache key based on document content and parameters
    doc_content_hash = hashlib.md5(str([doc.page_content for doc in docs]).encode()).hexdigest()
    cache_key = f"{doc_content_hash}_{chunk_size}_{overlap}"
    
    # Try to load from cache first
    cached_data = load_from_cache(f"chunks_{cache_key}")
    if cached_data is not None:
        print("✂️ Loaded chunks from cache")
        return cached_data
    
    print("✂️ Chunking text...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        add_start_index=True
    )
    chunks = splitter.split_documents(docs)
    
    # Save to cache
    save_to_cache(f"chunks_{cache_key}", chunks)
    print("💾 Saved chunks to cache")
    print(f"Generated {len(chunks)} chunks")
    return chunks

def raptor_clustering(chunks, levels=3):
    """Hierarchical clustering of document chunks with caching"""
    # Generate cache key based on chunk content and parameters
    chunk_content_hash = hashlib.md5(str([c.page_content for c in chunks]).encode()).hexdigest()
    cache_key = f"{chunk_content_hash}_{levels}"
    
    # Try to load from cache first
    cached_data = load_from_cache(f"raptor_{cache_key}")
    if cached_data is not None:
        print("🌳 Loaded RAPTOR clusters from cache")
        return cached_data
    
    print(f"🌳 Building RAPTOR hierarchy...")
    clustered_chunks = []
    embeddings = embed_text([c.page_content for c in chunks])
    print(f"Starting RAPTOR clustering for {len(chunks)} chunks")
    
    current_level = chunks
    for level in range(levels):
        if level == 0:
            clusters = current_level
        else:
            if len(current_level) <= 10:
                clusters = current_level
            else:
                kmeans = KMeans(n_clusters=min(10, len(current_level)))
                kmeans.fit(embeddings)
                new_clusters = []
                for cluster_idx in range(kmeans.n_clusters):
                    cluster_indices = [i for i, label in enumerate(kmeans.labels_) 
                                     if label == cluster_idx]
                    cluster_texts = [current_level[i].page_content for i in cluster_indices]
                    cluster_content = "\n\n".join(cluster_texts)
                    first_idx = cluster_indices[0]
                    metadata = current_level[first_idx].metadata.copy()
                    metadata["raptor_level"] = level
                    new_clusters.append(Document(page_content=cluster_content, metadata=metadata))
                clusters = new_clusters
        clustered_chunks.extend(clusters)
        current_level = clusters
    
    # Save to cache
    # Filter complex metadata to avoid issues with Chroma
    filtered_clustered_chunks = filter_complex_metadata(clustered_chunks)
    
    return filtered_clustered_chunks