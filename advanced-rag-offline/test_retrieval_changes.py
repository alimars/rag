#!/usr/bin/env python3
"""
Test script to verify that the retrieval changes work correctly.
This test checks that the retrieve method returns structured data with scores and metadata.
"""

import os
import sys
from pathlib import Path

# Add the rag_tool directory to the path so we can import its modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rag_tool"))

from rag_tool.pipeline import FocusedRAGPipeline
from rag_tool.indexing import MultiRepresentationIndex
from rag_tool.retrieval import RetrievalSystem
from langchain_core.documents import Document
import numpy as np

def test_retrieval_system():
    """Test the RetrievalSystem class with mock data"""
    print("🧪 Testing RetrievalSystem with mock data")
    print("=" * 50)
    
    try:
        # Create mock documents
        print("📄 Creating mock documents...")
        mock_docs = []
        for i in range(20):
            doc = Document(
                page_content=f"This is mock document {i} with some content for testing the retrieval system.",
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
        
        # Set up mock indexes (simplified for testing)
        index.documents = [doc.page_content for doc in mock_docs]
        
        print("✅ Mock indexes created successfully")
        
        # Create retrieval system
        print("\n🔍 Creating RetrievalSystem...")
        retriever = RetrievalSystem(index)
        
        # Mock the hybrid_search method to return mock results
        def mock_hybrid_search(query, top_k):
            # Return a subset of mock documents with mock scores
            # We'll create results with varying scores to test sorting
            results = []
            for i, doc in enumerate(mock_docs[:top_k]):
                # Generate a mock score (higher scores for higher i values to test sorting)
                score = float(top_k - i) / 10.0
                results.append((doc, score))
            return results
        
        # Replace the hybrid_search method with our mock
        index.hybrid_search = mock_hybrid_search
        
        print("✅ RetrievalSystem created successfully")
        
        # Test retrieve method
        print("\n🔍 Testing retrieve method...")
        query = "test query for retrieval system"
        top_k = 5
        
        print(f"   Query: {query}")
        print(f"   Requesting top {top_k} results")
        
        results = retriever.retrieve(query, top_k)
        print(f"   Received {len(results)} results")
        
        # Verify results format
        if not results:
            print("❌ No results returned")
            return False
            
        # Check that results are dictionaries with the expected keys
        for i, result in enumerate(results):
            if not isinstance(result, dict):
                print(f"❌ Result {i} is not a dictionary")
                return False
                
            expected_keys = {'content', 'score', 'metadata'}
            if not expected_keys.issubset(result.keys()):
                print(f"❌ Result {i} does not have expected keys: {expected_keys}")
                return False
                
            if not isinstance(result['content'], str):
                print(f"❌ Result {i} content is not a string")
                return False
                
            if not isinstance(result['score'], (int, float)):
                print(f"❌ Result {i} score is not a number")
                return False
                
            if not isinstance(result['metadata'], dict):
                print(f"❌ Result {i} metadata is not a dictionary")
                return False
                
            print(f"   Result {i+1}: Score={result['score']:.4f}, Content preview: {result['content'][:50]}...")
            
            # Check for metadata fields
            if 'page' in result['metadata']:
                print(f"     Page: {result['metadata']['page']}")
            if 'chunk_id' in result['metadata']:
                print(f"     Chunk ID: {result['metadata']['chunk_id']}")
        
        # Test that results are sorted by score in descending order
        print("\n🔍 Testing score sorting...")
        scores = [result['score'] for result in results]
        if scores != sorted(scores, reverse=True):
            print("❌ Results are not sorted by score in descending order")
            return False
        else:
            print("✅ Results are correctly sorted by score in descending order")
        
        # Test with different top_k values
        print(f"\n🔍 Testing retrieve with top_k=10...")
        results2 = retriever.retrieve(query, 10)
        print(f"   Received {len(results2)} results (requested 10)")
        
        if len(results2) != 10:
            print(f"❌ Expected 10 results, but got {len(results2)}")
            return False
            
        # Test that results2 are also sorted by score in descending order
        scores2 = [result['score'] for result in results2]
        if scores2 != sorted(scores2, reverse=True):
            print("❌ Results2 are not sorted by score in descending order")
            return False
        else:
            print("✅ Results2 are correctly sorted by score in descending order")
            
        print("\n" + "=" * 50)
        print("🎉 All RetrievalSystem tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_pipeline_integration():
    """Test the pipeline integration with the updated retrieval system"""
    print("\n\n🧪 Testing Pipeline Integration")
    print("=" * 50)
    
    try:
        # Create a simple pipeline with mock data
        print("📄 Creating mock pipeline...")
        
        # Create a mock pipeline class that doesn't require actual documents
        class MockPipeline(FocusedRAGPipeline):
            def initialize(self):
                # Skip actual initialization and just set up the mock retriever
                print("🔄 Initializing mock pipeline...")
                
                # Create mock index
                self.index = MultiRepresentationIndex()
                self.index.documents = [f"Mock document {i}" for i in range(10)]
                
                # Create retriever
                self.retriever = RetrievalSystem(self.index)
                
                # Mock the hybrid_search method
                def mock_hybrid_search(query, top_k):
                    mock_docs = []
                    for i in range(min(top_k, 10)):
                        doc = Document(
                            page_content=f"Mock document {i} content for query: {query}",
                            metadata={
                                "source": f"mock_document_{i}.txt",
                                "page": i % 3 + 1,
                                "chunk_id": f"chunk_{i}"
                            }
                        )
                        # Generate scores in a way that tests sorting
                        mock_docs.append((doc, float(top_k - i) / 10.0))
                    return mock_docs
                
                self.index.hybrid_search = mock_hybrid_search
                self.is_initialized = True
                print("✅ Mock pipeline initialized successfully")
                return True
        
        # Create and initialize mock pipeline
        pipeline = MockPipeline("mock_data_path")
        pipeline.initialize()
        
        # Test query method
        print("\n🔍 Testing pipeline query method...")
        query = "test query for pipeline"
        
        result = pipeline.query(query, return_original=True)
        
        # Verify result structure
        expected_keys = {'original_response', 'detailed_results', 'translation', 'source_language'}
        if not expected_keys.issubset(result.keys()):
            print(f"❌ Result does not have expected keys: {expected_keys}")
            return False
            
        print(f"✅ Query result has expected structure")
        print(f"   Original response length: {len(result['original_response'])} characters")
        print(f"   Detailed results count: {len(result['detailed_results'])}")
        print(f"   Source language: {result['source_language']}")
        
        # Check detailed results
        if result['detailed_results']:
            first_result = result['detailed_results'][0]
            expected_detail_keys = {'doc_name', 'page', 'chunk_id', 'similarity_score', 'content'}
            if not expected_detail_keys.issubset(first_result.keys()):
                print(f"❌ Detailed result does not have expected keys: {expected_detail_keys}")
                return False
            print(f"✅ Detailed results have expected structure")
            print(f"   First result document: {first_result['doc_name']}")
            print(f"   First result score: {first_result['similarity_score']}")
            print(f"   First result page: {first_result['page']}")
            
            # Test that detailed results are sorted by score in descending order
            scores = [result['similarity_score'] for result in result['detailed_results']]
            if scores != sorted(scores, reverse=True):
                print("❌ Detailed results are not sorted by score in descending order")
                return False
            else:
                print("✅ Detailed results are correctly sorted by score in descending order")
                
            # Test that page metadata is properly handled (not showing as 'Unknown')
            pages = [result['page'] for result in result['detailed_results']]
            if 'Unknown' in pages:
                print("❌ Page metadata is showing as 'Unknown'")
                return False
            else:
                print("✅ Page metadata is properly handled")
            
        print("\n" + "=" * 50)
        print("🎉 All Pipeline Integration tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline integration test failed with error: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("Retrieval System Changes Test")
    print("=" * 40)
    
    success1 = test_retrieval_system()
    success2 = test_pipeline_integration()
    
    print("\n" + "=" * 40)
    if success1 and success2:
        print("🎉 All tests passed!")
        print("✅ Retrieval system changes work correctly")
    else:
        print("❌ Some tests failed.")
        sys.exit(1)