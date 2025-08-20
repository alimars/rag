#!/usr/bin/env python3
"""
Simple test script to verify that the hybrid_search method works correctly after changes.
"""

import os
import sys
from pathlib import Path

# Add the rag_tool directory to the path so we can import its modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rag_tool"))

from rag_tool.document_processor import load_documents, chunk_text, raptor_clustering
from rag_tool.indexing import MultiRepresentationIndex

def test_hybrid_search():
    """Test the hybrid_search method"""
    print("🧪 Testing hybrid_search method")
    print("=" * 40)
    
    try:
        # Set up environment
        docs_path = os.path.join(os.path.dirname(__file__), "documents")
        language = "en"  # Use English for simplicity
        
        print(f"📁 Documents path: {docs_path}")
        
        # Check if documents exist
        if not os.path.exists(docs_path):
            print("❌ Documents directory not found")
            return False
            
        pdf_files = [f for f in os.listdir(docs_path) if f.endswith('.pdf')]
        if not pdf_files:
            print("❌ No PDF documents found in documents directory")
            return False
            
        print(f"📄 Found PDF documents: {pdf_files[:3]}...")  # Show first 3
        
        # Load and process documents
        print("\n1️⃣  Loading and processing documents...")
        docs = load_documents(docs_path, language)
        print(f"📄 Processed {len(docs)} documents")
        
        if len(docs) == 0:
            print("❌ No documents loaded")
            return False
            
        # Chunk text
        print("\n2️⃣  Chunking text...")
        chunks = chunk_text(docs)
        print(f"✂️  Created {len(chunks)} chunks")
        
        if len(chunks) == 0:
            print("❌ No chunks created")
            return False
            
        # Create RAPTOR clustering
        print("\n3️⃣  Building RAPTOR hierarchy...")
        raptor_chunks = raptor_clustering(chunks)
        print(f"🌲 Built {len(raptor_chunks)} hierarchical clusters")
        
        # Build indexes
        print("\n4️⃣  Building indexes...")
        index = MultiRepresentationIndex()
        index.build_indexes(chunks, raptor_chunks)
        print("🏗️  Indexes built successfully")
        
        # Test hybrid search
        print("\n5️⃣  Testing hybrid_search method...")
        query = "What are the main topics?"
        top_k = 5
        
        print(f"🔍 Query: {query}")
        print(f"📊 Requesting top {top_k} results")
        
        results = index.hybrid_search(query, top_k)
        print(f"📄 Received {len(results)} results")
        
        # Verify results format
        if not results:
            print("❌ No results returned")
            return False
            
        # Check that results are tuples of (document, score)
        for i, result in enumerate(results):
            if not isinstance(result, tuple) or len(result) != 2:
                print(f"❌ Result {i} is not a tuple of (document, score)")
                return False
                
            doc, score = result
            if not hasattr(doc, 'page_content'):
                print(f"❌ Result {i} document does not have page_content attribute")
                return False
                
            if not isinstance(score, (int, float)):
                print(f"❌ Result {i} score is not a number")
                return False
                
            print(f"   Result {i+1}: Score={score:.4f}, Content preview: {doc.page_content[:100]}...")
            
        # Test with top_k=10
        print(f"\n6️⃣  Testing hybrid_search with top_k=10...")
        results2 = index.hybrid_search(query, 10)
        print(f"📄 Received {len(results2)} results (requested 10)")
        
        if len(results2) > 10:
            print("❌ Returned more results than requested")
            return False
            
        print("✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("Hybrid Search Method Test")
    print("=" * 25)
    
    success = test_hybrid_search()
    
    print("\n" + "=" * 25)
    if success:
        print("🎉 All tests passed!")
        print("✅ hybrid_search method works correctly")
    else:
        print("❌ Some tests failed.")
        sys.exit(1)