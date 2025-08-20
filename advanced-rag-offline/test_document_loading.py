#!/usr/bin/env python3
"""
Test script to debug document loading issues
"""
import os
import sys

# Add the rag_tool directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'rag_tool'))

from rag_tool.document_processor import load_documents

def test_document_loading():
    """Test document loading with detailed output"""
    print("Testing document loading...")
    print("Documents directory contents:")
    docs_dir = "./documents"
    if os.path.exists(docs_dir):
        for file in os.listdir(docs_dir):
            file_path = os.path.join(docs_dir, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                print(f"  {file} ({size} bytes)")
    else:
        print(f"Documents directory {docs_dir} does not exist")
        return
    
    print("\nLoading documents...")
    docs = load_documents('./documents', 'ar')
    print(f"Loaded {len(docs)} documents")
    
    if docs:
        print("\nDocument details:")
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Unknown')
            content_length = len(doc.page_content)
            print(f"  Document {i+1}:")
            print(f"    Source: {source}")
            print(f"    Content length: {content_length} characters")
            if content_length > 0:
                # Show first 200 characters of content
                preview = doc.page_content[:200].replace('\n', '\\n')
                print(f"    Content preview: {preview}")
                if content_length > 200:
                    print(f"    ... (and {content_length - 200} more characters)")
            else:
                print("    Content: EMPTY")
            print()

if __name__ == "__main__":
    test_document_loading()