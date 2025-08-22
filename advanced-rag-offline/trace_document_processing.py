#!/usr/bin/env python3
"""
Trace script to understand document processing flow
"""
import os
import sys

# Add the rag_tool directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'rag_tool'))

from rag_tool.document_processor import load_documents, chunk_text, raptor_clustering

def trace_document_processing():
    """Trace document processing with detailed output"""
    print("=== TRACING DOCUMENT PROCESSING ===")
    
    print("\n1. Loading documents...")
    docs = load_documents('./documents', 'ar')
    print(f"   Loaded {len(docs)} documents")
    
    # Check for duplicate sources
    sources = [doc.metadata.get('source', 'Unknown') for doc in docs]
    unique_sources = list(set(sources))
    print(f"   Unique document sources: {len(unique_sources)}")
    print(f"   All sources: {sources}")
    
    if len(unique_sources) != len(sources):
        print("   WARNING: Duplicate document sources detected!")
        for source in unique_sources:
            count = sources.count(source)
            if count > 1:
                print(f"     {source}: {count} times")
    
    print("\n2. Analyzing document content...")
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'Unknown')
        content_length = len(doc.page_content)
        print(f"   Document {i+1} ({source}): {content_length} characters")
    
    print("\n3. Chunking text...")
    chunks = chunk_text(docs)
    print(f"   Created {len(chunks)} chunks")
    
    print("\n4. Analyzing chunks...")
    # Check for duplicate chunk IDs
    chunk_ids = [chunk.metadata.get('chunk_id', f'unknown_{i}') for i, chunk in enumerate(chunks)]
    unique_chunk_ids = list(set(chunk_ids))
    print(f"   Unique chunk IDs: {len(unique_chunk_ids)}")
    print(f"   Total chunk IDs: {len(chunk_ids)}")
    
    if len(unique_chunk_ids) != len(chunk_ids):
        print("   WARNING: Duplicate chunk IDs detected!")
        for chunk_id in unique_chunk_ids:
            count = chunk_ids.count(chunk_id)
            if count > 1:
                print(f"     {chunk_id}: {count} times")
    
    # Check chunk sources
    chunk_sources = [chunk.metadata.get('source', 'Unknown') for chunk in chunks]
    unique_chunk_sources = list(set(chunk_sources))
    print(f"   Unique chunk sources: {len(unique_chunk_sources)}")
    print(f"   All chunk sources: {unique_chunk_sources}")
    
    # Count chunks per source
    source_chunk_counts = {}
    for source in chunk_sources:
        source_chunk_counts[source] = source_chunk_counts.get(source, 0) + 1
    
    print("\n5. Chunk count per source:")
    for source, count in source_chunk_counts.items():
        print(f"   {source}: {count} chunks")
    
    print("\n6. Performing RAPTOR clustering...")
    raptor_chunks = raptor_clustering(chunks)
    print(f"   Created {len(raptor_chunks)} RAPTOR clusters")
    
    print(f"\n=== SUMMARY ===")
    print(f"Documents loaded: {len(docs)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"RAPTOR clusters created: {len(raptor_chunks)}")
    print(f"Total chunks (original + RAPTOR): {len(chunks) + len(raptor_chunks)}")

if __name__ == "__main__":
    trace_document_processing()