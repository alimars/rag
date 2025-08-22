#!/usr/bin/env python3
"""
Simple test to understand what UnstructuredLoader is doing
"""
import os
import sys

# Add the rag_tool directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'rag_tool'))

from langchain_unstructured import UnstructuredLoader
from pathlib import Path

def test_unstructured_loader():
    """Test UnstructuredLoader with a single document"""
    print("Testing UnstructuredLoader...")
    docs_dir = "./documents"
    if os.path.exists(docs_dir):
        for file in os.listdir(docs_dir):
            if file.endswith('.pdf'):
                file_path = os.path.join(docs_dir, file)
                print(f"\nLoading {file}...")
                try:
                    loader = UnstructuredLoader(str(file_path))
                    docs = loader.load()
                    print(f"  Loaded {len(docs)} document parts")
                    for i, doc in enumerate(docs[:3]):  # Show first 3 parts
                        content_length = len(doc.page_content)
                        print(f"    Part {i+1}: {content_length} characters")
                        if content_length > 0:
                            # Show first 200 characters of content
                            preview = doc.page_content[:200].replace('\n', '\\n')
                            print(f"      Content preview: {preview}")
                            if content_length > 200:
                                print(f"      ... (and {content_length - 200} more characters)")
                    if len(docs) > 3:
                        print(f"      ... and {len(docs) - 3} more parts")
                except Exception as e:
                    print(f"  Error loading {file}: {str(e)}")
    else:
        print(f"Documents directory {docs_dir} does not exist")

if __name__ == "__main__":
    test_unstructured_loader()