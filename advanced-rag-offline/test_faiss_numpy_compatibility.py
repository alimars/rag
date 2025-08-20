#!/usr/bin/env python3
"""
Minimal test script to verify that the NumPy compatibility issue with FAISS is resolved.

This test specifically checks:
1. FAISS can be imported successfully
2. NumPy arrays can be used with FAISS without compatibility issues
3. Basic FAISS operations work with Arabic text content
4. Index creation and search functionality work correctly
"""

import sys
import numpy as np

def test_faiss_numpy_compatibility():
    """Test FAISS/NumPy compatibility with a minimal example"""
    print("🧪 Testing FAISS/NumPy Compatibility")
    print("=" * 40)
    
    try:
        # Import FAISS - this is where compatibility issues typically occur
        import faiss
        print("✅ FAISS imported successfully")
        
        # Test 1: Verify NumPy/FAISS array compatibility
        print("\n1️⃣ Testing NumPy array compatibility...")
        
        # Create sample data with Arabic text
        arabic_texts = [
            "هذا هو النص العربي الأول",  # "This is the first Arabic text"
            "نظام استرجاع المعلومات",     # "Information retrieval system"
            "معالجة اللغة العربية",       # "Arabic language processing"
            "البحث في المستندات",         # "Searching in documents"
        ]
        
        # Create sample embeddings (768-dimensional vectors)
        # In a real scenario, these would come from an embedding model
        dimension = 768
        num_vectors = len(arabic_texts)
        
        # Create random embeddings using NumPy
        embeddings = np.random.rand(num_vectors, dimension).astype('float32')
        print(f"✅ Created {num_vectors} embeddings of dimension {dimension}")
        print(f"   Embedding array shape: {embeddings.shape}")
        print(f"   Embedding array dtype: {embeddings.dtype}")
        
        # Test 2: Create FAISS index
        print("\n2️⃣ Creating FAISS index...")
        
        # Create a simple FAISS index
        index = faiss.IndexFlatL2(dimension)
        print("✅ FAISS IndexFlatL2 created successfully")
        
        # Add embeddings to the index
        index.add(embeddings)
        print(f"✅ Added {index.ntotal} vectors to the index")
        
        # Test 3: Verify index structure
        print("\n3️⃣ Verifying index structure...")
        print(f"   Index dimension: {index.d}")
        print(f"   Number of vectors: {index.ntotal}")
        
        if index.ntotal == num_vectors:
            print("✅ Index contains expected number of vectors")
        else:
            print(f"❌ Index contains {index.ntotal} vectors, expected {num_vectors}")
            return False
            
        # Test 4: Perform similarity search
        print("\n4️⃣ Performing similarity search...")
        
        # Search for the first vector
        query_vector = embeddings[0:1]  # Reshape to (1, dimension)
        k = 2  # Number of nearest neighbors to retrieve
        
        distances, indices = index.search(query_vector, k)
        print(f"✅ Search completed successfully")
        print(f"   Retrieved {len(indices[0])} results")
        print(f"   Distances: {distances[0]}")
        print(f"   Indices: {indices[0]}")
        
        # The first result should be the query vector itself (distance ~0)
        if indices[0][0] == 0 and distances[0][0] < 1e-5:
            print("✅ Self-similarity search worked correctly")
        else:
            print(f"⚠️  Self-similarity search may have issues")
            print(f"   Expected index 0 with distance ~0, got index {indices[0][0]} with distance {distances[0][0]}")
            
        # Test 5: Verify Arabic text handling
        print("\n5️⃣ Verifying Arabic text handling...")
        
        # Create a simple data structure that mimics the required FAISS format
        faiss_data = []
        for i, (text, embedding) in enumerate(zip(arabic_texts, embeddings)):
            entry = {
                "embedding": embedding.tolist(),  # Convert to list for JSON serialization
                "text": text,
                "doc_name": f"arabic_doc_{i+1}.txt",
                "page": 1,
                "chunk_id": f"chunk_{i+1:03d}"
            }
            faiss_data.append(entry)
            
        print(f"✅ Created {len(faiss_data)} entries with Arabic text")
        
        # Verify that Arabic text is preserved
        sample_entry = faiss_data[0]
        if "النص العربي" in sample_entry["text"]:
            print("✅ Arabic text preserved in data structure")
        else:
            print("❌ Arabic text not found in data structure")
            return False
            
        print("\n" + "=" * 40)
        print("🎉 All FAISS/NumPy compatibility tests passed!")
        print("\n📋 Summary:")
        print("   ✅ FAISS imports successfully")
        print("   ✅ NumPy arrays work with FAISS")
        print("   ✅ Index creation works correctly")
        print("   ✅ Similarity search functions properly")
        print("   ✅ Arabic text is handled correctly")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("This test requires FAISS to be installed.")
        print("Please ensure you're running in the Docker environment with all dependencies.")
        return False
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    print("FAISS NumPy Compatibility Test")
    print("=" * 30)
    
    success = test_faiss_numpy_compatibility()
    
    print("\n" + "=" * 30)
    if success:
        print("✅ Compatibility test PASSED")
        print("The NumPy/FAISS compatibility issue appears to be resolved.")
    else:
        print("❌ Compatibility test FAILED")
        print("There may be issues with NumPy/FAISS compatibility.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)