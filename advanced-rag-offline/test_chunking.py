#!/usr/bin/env python3
"""
Test script to debug document chunking issues
"""
import os
import sys

# Add the rag_tool directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'rag_tool'))

from rag_tool.document_processor import load_documents, chunk_text, raptor_clustering

def test_document_chunking():
    """Test document chunking with detailed output"""
    print("Testing document chunking...")
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
    
    print("\nChunking text...")
    chunks = chunk_text(docs)
    print(f"Created {len(chunks)} chunks")
    
    print("\nPerforming RAPTOR clustering...")
    raptor_chunks = raptor_clustering(chunks)
    print(f"Created {len(raptor_chunks)} RAPTOR clusters")
    
    print(f"\nTotal chunks (original + RAPTOR): {len(chunks) + len(raptor_chunks)}")
    
    # Analyze chunk sources
    if chunks:
        print("\nChunk source analysis:")
        source_counts = {}
        for chunk in chunks:
            source = chunk.metadata.get('source', 'Unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
        for source, count in source_counts.items():
            print(f"  {source}: {count} chunks")
    
    if raptor_chunks:
        print("\nRAPTOR cluster analysis:")
        level_counts = {}
        for chunk in raptor_chunks:
            level = chunk.metadata.get('raptor_level', 0)
            level_counts[level] = level_counts.get(level, 0) + 1
        for level, count in sorted(level_counts.items()):
            print(f"  Level {level}: {count} clusters")

if __name__ == "__main__":
    test_document_chunking()