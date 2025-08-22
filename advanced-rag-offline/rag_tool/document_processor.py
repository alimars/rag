import pytesseract
from pdf2image import convert_from_path
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
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

# Cache directory
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_key(path: str, language: str = "en") -> str:
    """Generate a cache key based on path and language"""
    try:
        dir_path = Path(path).resolve()  # Normalize to absolute path
        dir_mtime = dir_path.stat().st_mtime
        
        # Get file list and modification times for more accurate change detection
        file_stats = [
            (f.name, f.stat().st_mtime)
            for f in sorted(dir_path.rglob('*'))
            if f.is_file()
        ]
        file_stats.sort(key=lambda x: x[0])  # Sort by filename for consistent ordering
        file_hash = hashlib.md5(str(file_stats).encode()).hexdigest()
    except OSError:
        dir_mtime = time.time()
        file_hash = "error"
    
    # Combine both directory mtime and file list hash for better change detection
    hash_input = f"{str(dir_path)}_{language}_{int(dir_mtime)}_{file_hash}"
    
    
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
                data = pickle.load(f)
                
                return data
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
    tesseract_lang = LANGUAGE_MAP.get(language, "ara")
    images = convert_from_path(file_path)
    full_text = ""
    for i, img in enumerate(images):
        text = pytesseract.image_to_string(img, lang=tesseract_lang)
        full_text += f"Page {i+1}:\n{text}\n\n"
    return full_text

def load_documents(path, language="ar"):
    """Load and process documents from directory with caching"""
    # Normalize path and generate cache key
    path = str(Path(path).resolve())  # Ensure consistent path format
    cache_key = get_cache_key(path, language)
    cache_file_key = f"documents_{cache_key}"
    
    # Try to load from cache first
    cached_data = load_from_cache(cache_file_key)
    if cached_data is not None:
        print("📄 Loaded documents from cache")
        return cached_data
    else:
        
        print("🔄 Loading and processing documents...")
        tesseract_lang = LANGUAGE_MAP.get(language, "ara")
        loaders = {
            '.pdf': UnstructuredLoader,
            '.docx': UnstructuredLoader,
            '.doc': UnstructuredLoader,
            '.txt': UnstructuredLoader
        }
        documents = []
        
        import concurrent.futures
        import time
        
        # Process files with timeout
        file_paths = list(sorted(Path(path).rglob('*')))
        total_files = len([fp for fp in file_paths if fp.suffix.lower() in loaders])
        processed_files = 0
        
        # Define process_file function outside the loop
        def process_file(file_path):
            try:
                if file_path.suffix.lower() == '.pdf':
                    try:
                        loader = UnstructuredLoader(str(file_path))
                        docs = loader.load()
                        if docs and docs[0].page_content.strip():
                            # Combine all parts into a single document
                            combined_content = "\n\n".join([doc.page_content for doc in docs if doc.page_content.strip()])
                            combined_metadata = docs[0].metadata.copy()
                            combined_metadata["source"] = str(file_path)
                            return [Document(page_content=combined_content, metadata=combined_metadata)]
                    except Exception as e:
                        print(f"UnstructuredLoader failed for {file_path}: {str(e)}")
                        pass
                    text = ocr_pdf(str(file_path), language)
                    metadata = {"source": str(file_path)}
                    return [Document(page_content=text, metadata=metadata)]
                else:
                    loader = UnstructuredLoader(str(file_path))
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["source"] = str(file_path)
                    return docs
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                # Return empty list to continue with other files
                return []
        
        for file_path in file_paths:
            if file_path.suffix.lower() in loaders:
                processed_files += 1
                print(f"Processing file {processed_files}/{total_files}: {file_path.name}")
                
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
            save_to_cache(cache_file_key, documents)
            print("💾 Saved documents to cache")
            
        except Exception as e:
            print(f"Warning: Could not save documents to cache: {str(e)}")
        
        return documents

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
    
    # Add page_id and chunk_id to metadata
    for i, chunk in enumerate(chunks):
        # Extract page number from metadata if available
        page_number = chunk.metadata.get('page', 1)
        
        # Add page_id to metadata
        chunk.metadata['page'] = page_number
        
        # Add chunk_id to metadata
        source = chunk.metadata.get('source', 'unknown')
        # Extract filename without extension
        filename = os.path.basename(source)
        filename_without_ext = os.path.splitext(filename)[0]
        chunk.metadata['chunk_id'] = f"{filename_without_ext}_chunk_{i:04d}"
    
    # Save to cache
    save_to_cache(f"chunks_{cache_key}", chunks)
    print("💾 Saved chunks to cache")
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
    embeddings = embed_text([c.page_content for c in chunks])
    print(f"Starting RAPTOR clustering for {len(chunks)} chunks")
    
    current_level = chunks
    for level in range(levels):
        if level == 0:
            # For level 0, we keep the original chunks
            clusters = current_level
        else:
            # For subsequent levels, we cluster the previous level
            if len(current_level) <= 10:
                # If we have 10 or fewer chunks, don't cluster further
                clusters = current_level
                break
            else:
                # Perform clustering
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
        current_level = clusters
    
    # Save to cache - only return the final level of clusters
    save_to_cache(f"raptor_{cache_key}", current_level)
    print("💾 Saved RAPTOR clusters to cache")
    return current_level