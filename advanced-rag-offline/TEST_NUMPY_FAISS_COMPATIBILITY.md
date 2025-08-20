# NumPy/FAISS Compatibility Test

This document provides instructions for running the NumPy/FAISS compatibility test to verify that the compatibility issue has been resolved.

## Test Overview

The test script `test_faiss_numpy_compatibility.py` verifies that:
1. FAISS can be imported successfully without NumPy compatibility issues
2. NumPy arrays can be used with FAISS without errors
3. Basic FAISS operations (index creation, similarity search) work correctly
4. Arabic text content is handled properly in the FAISS data structures

## Prerequisites

- Docker and Docker Compose installed
- Ollama running on the host machine (for embedding models)

## Running the Test

### Method 1: Run directly in the Docker container

1. Start the RAG service using docker-compose:
   ```bash
   docker-compose up -d rag-api
   ```

2. Execute the test script inside the running container:
   ```bash
   docker exec -it rag python test_faiss_numpy_compatibility.py
   ```

### Method 2: Run as a one-off command

```bash
docker-compose run --rm rag-api python test_faiss_numpy_compatibility.py
```

### Method 3: Run during container build (for development)

If you want to run the test as part of the container startup process, you can temporarily modify the `start_rag.sh` script or Dockerfile to execute the test before starting the main service.

## Expected Output

A successful test run will show output similar to:

```
FAISS NumPy Compatibility Test
==============================

🧪 Testing FAISS/NumPy Compatibility
========================================

✅ FAISS imported successfully

1️⃣ Testing NumPy array compatibility...
✅ Created 4 embeddings of dimension 768
   Embedding array shape: (4, 768)
   Embedding array dtype: float32

2️⃣ Creating FAISS index...
✅ FAISS IndexFlatL2 created successfully
✅ Added 4 vectors to the index

3️⃣ Verifying index structure...
   Index dimension: 768
   Number of vectors: 4
✅ Index contains expected number of vectors

4️⃣ Performing similarity search...
✅ Search completed successfully
   Retrieved 2 results
   Distances: [0.0000000e+00 2.9745650e+02]
   Indices: [0 2]
✅ Self-similarity search worked correctly

5️⃣ Verifying Arabic text handling...
✅ Created 4 entries with Arabic text
✅ Arabic text preserved in data structure

========================================
🎉 All FAISS/NumPy compatibility tests passed!

📋 Summary:
   ✅ FAISS imports successfully
   ✅ NumPy arrays work with FAISS
   ✅ Index creation works correctly
   ✅ Similarity search functions properly
   ✅ Arabic text is handled correctly

==============================
✅ Compatibility test PASSED
The NumPy/FAISS compatibility issue appears to be resolved.
```

## Troubleshooting

### ImportError: No module named 'faiss'

This indicates that FAISS is not installed in the environment. Ensure you're running the test inside the Docker container where all dependencies are installed.

### NumPy/FAISS dtype errors

If you see errors related to data types (e.g., "cannot convert float64 to float32"), this indicates a NumPy compatibility issue. The test script ensures proper dtype handling by using:
```python
embeddings = np.random.rand(num_vectors, dimension).astype('float32')
```

### Arabic text encoding issues

If Arabic text appears garbled, check that JSON serialization uses `ensure_ascii=False`:
```python
json.dumps(data, ensure_ascii=False)
```

## Test Components

The test script verifies several key components:

1. **FAISS Import**: Ensures FAISS can be imported without compatibility issues
2. **NumPy Array Creation**: Creates sample embeddings using NumPy
3. **Index Creation**: Builds a FAISS index with the embeddings
4. **Similarity Search**: Performs a basic search operation
5. **Arabic Text Handling**: Verifies Arabic content is preserved in data structures

## Related Files

- `test_faiss_numpy_compatibility.py` - Main test script
- `requirements.txt` - Specifies FAISS and NumPy versions
- `Dockerfile` - Container build configuration
- `docker-compose.yml` - Service configuration