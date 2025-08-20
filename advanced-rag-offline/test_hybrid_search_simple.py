#!/usr/bin/env python3
"""
Simple test script to verify that the hybrid_search method works correctly after changes.
This test uses mock data to avoid dependency on Ollama and document processing.
"""

import os
import sys
from pathlib import Path

# Add the rag_tool directory to the path so we can import its modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rag_tool"))

from rag_tool.indexing import MultiRepresentationIndex
from langchain_core.documents import Document
import numpy as np

class MockFAISS:
    """Mock FAISS class for testing"""
    def __init__(self, documents):
        self.documents = documents
    
    def similarity_search_with_score(self, query, k=10):
        # Return a subset of documents with mock scores
        # In real FAISS, lower scores are better (distance), so we'll simulate that
        results = []
        for i, doc in enumerate(self.documents[:k]):
            # Generate a mock score (lower is better)
            score = float(i + 1) / 10.0
            results.append((doc, score))
        return results

class MockBM25Okapi:
    """Mock BM25Okapi class for testing"""
    def __init__(self, tokenized_docs):
        self.tokenized_docs = tokenized_docs
    
    def get_scores(self, tokenized_query):
        # Return mock scores for each document
        return np.array([1.0 / (i + 1) for i in range(len(self.tokenized_docs))])

def test_hybrid_search():
    """Test the hybrid_search method with mock data"""
    print("🧪 Testing hybrid_search method with mock data")
    print("=" * 50)
    
    try:
        # Create mock documents
        print("📄 Creating mock documents...")
        mock_docs = []
        for i in range(20):
            doc = Document(
                page_content=f"This is mock document {i} with some content for testing the hybrid search functionality.",
                metadata={
                    "source": f"mock_document_{i}.txt",
                    "page": i % 5 + 1,
                    "chunk_id": f"chunk_{i}"
                }
            )
            mock_docs.append(doc)
        
        print(f"✅ Created {len(mock_docs)} mock documents")
        
        # Create mock indexes
        print("\n🏗️  Creating mock indexes...")
        index = MultiRepresentationIndex()
        
        # Set up mock indexes
        index.dense_index = MockFAISS(mock_docs)
        index.raptor_index = MockFAISS(mock_docs[:10])  # Smaller set for RAPTOR
        index.documents = [doc.page_content for doc in mock_docs]
        index.sparse_index = MockBM25Okapi([doc.page_content.split() for doc in mock_docs])
        
        print("✅ Mock indexes created successfully")
        
        # Test hybrid search
        print("\n🔍 Testing hybrid_search method...")
        query = "test query for hybrid search"
        top_k = 5
        
        print(f"   Query: {query}")
        print(f"   Requesting top {top_k} results")
        
        results = index.hybrid_search(query, top_k)
        print(f"   Received {len(results)} results")
        
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
                
            print(f"   Result {i+1}: Score={score:.4f}, Content preview: {doc.page_content[:50]}...")
            
        # Test that exactly top_k results are returned
        if len(results) != top_k:
            print(f"❌ Expected {top_k} results, but got {len(results)}")
            return False
            
        # Test with different top_k values
        print(f"\n🔍 Testing hybrid_search with top_k=10...")
        results2 = index.hybrid_search(query, 10)
        print(f"   Received {len(results2)} results (requested 10)")
        
        if len(results2) != 10:
            print(f"❌ Expected 10 results, but got {len(results2)}")
            return False
            
        # Test error handling by creating an index with None values
        print("\n🛡️  Testing error handling...")
        index_with_errors = MultiRepresentationIndex()
        index_with_errors.dense_index = None
        index_with_errors.raptor_index = None
        index_with_errors.documents = []
        index_with_errors.sparse_index = None
        
        # This should not crash but return an empty list
        error_results = index_with_errors.hybrid_search(query, 5)
        print(f"   Error handling test: Returned {len(error_results)} results with None indexes")
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("Hybrid Search Method Test (Simplified)")
    print("=" * 40)
    
    success = test_hybrid_search()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 All tests passed!")
        print("✅ hybrid_search method works correctly")
    else:
        print("❌ Some tests failed.")
        sys.exit(1)