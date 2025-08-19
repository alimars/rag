#!/usr/bin/env python3
"""
Test script to verify that the caching fix is working correctly
"""

import os
import sys
import time
import shutil
from pathlib import Path

# Add the rag_tool directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'advanced-rag-offline'))

from rag_tool.pipeline import FocusedRAGPipeline

def test_caching():
    """Test that caching is working correctly"""
    print("Testing caching functionality...")
    
    # Create a test documents directory
    test_docs_dir = os.path.join(os.path.dirname(__file__), 'test_documents')
    os.makedirs(test_docs_dir, exist_ok=True)
    
    # Create a test document
    test_doc_path = os.path.join(test_docs_dir, 'test.txt')
    with open(test_doc_path, 'w', encoding='utf-8') as f:
        f.write("This is a test document for caching functionality.")
    
    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = FocusedRAGPipeline(test_docs_dir, "en")
    
    # Time the first initialization
    start_time = time.time()
    pipeline.initialize()
    first_init_time = time.time() - start_time
    print(f"First initialization took {first_init_time:.2f} seconds")
    
    # Check if cache was created
    cache_dir = os.path.join(os.path.dirname(__file__), 'advanced-rag-offline', 'cache')
    if os.path.exists(cache_dir):
        cache_files = list(Path(cache_dir).rglob('*'))
        print(f"Cache directory contains {len(cache_files)} files")
        for file in cache_files:
            print(f"  - {file.name}")
    
    # Initialize a second pipeline to test cache loading
    print("\nInitializing second pipeline...")
    pipeline2 = FocusedRAGPipeline(test_docs_dir, "en")
    
    # Time the second initialization
    start_time = time.time()
    pipeline2.initialize()
    second_init_time = time.time() - start_time
    print(f"Second initialization took {second_init_time:.2f} seconds")
    
    # Check if caching is working (second init should be faster)
    if second_init_time < first_init_time * 0.5:
        print("✅ Caching is working correctly - second initialization was significantly faster")
    else:
        print("⚠️  Caching may not be working optimally - second initialization was not much faster")
    
    # Clean up test documents
    shutil.rmtree(test_docs_dir, ignore_errors=True)
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_caching()