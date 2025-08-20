# FAISS Arabic Content Verification Tests

This directory contains tests to verify that the FAISS implementation works correctly with Arabic content.

## Test Files

1. **`test_faiss_arabic_basic.py`** - Basic test that verifies Arabic text encoding and FAISS structure without external dependencies
2. **`test_faiss_arabic_integration.py`** - Integration test designed to run within the Docker environment

## Running the Tests

### Basic Test (No Dependencies Required)

```bash
cd advanced-rag-offline
python test_faiss_arabic_basic.py
```

This test verifies:
- Arabic text encoding in data structures
- FAISS index structure compliance
- JSON serialization with Arabic text
- Search simulation with Arabic queries

### Integration Test (Docker Environment)

To run the integration test with all dependencies:

```bash
# Build and start the Docker environment
docker-compose up -d

# Run the integration test inside the container
docker-compose exec rag-api python test_faiss_arabic_integration.py
```

## Test Results Summary

The basic test verifies that:

✅ **Arabic Text Encoding**: Arabic Unicode characters (U+0600-U+06FF) are properly handled
✅ **FAISS Index Structure**: Index entries contain the required fields:
   - `embedding`: Vector representation
   - `text`: Chunk text content
   - `doc_name`: Source document filename
   - `page`: Page number
   - `chunk_id`: Unique chunk identifier
✅ **Retrieval Functionality**: Search works with Arabic queries
✅ **JSON Serialization**: Arabic text is preserved when serializing to JSON

## FAISS Index Structure

The FAISS implementation creates indexes with the following structure:

```json
{
  "embedding": [0.123, 0.456, 0.789],
  "text": "هذا هو النص العربي للتجريب",
  "doc_name": "arabic_document_1.txt",
  "page": 1,
  "chunk_id": "chunk_001"
}