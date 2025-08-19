# Offline RAG Pipeline with Translation

Private RAG system with advanced features running entirely offline.

## Technologies
- langchain-ollama
- langchain-unstructured
- Ollama
- Unstructured.io

## Features
- Multi-Query Generation
- RAG-Fusion retrieval
- Query Decomposition
- Hierarchical Indexing (RAPTOR)
- Dedicated Re-ranking
- Offline Translation (mistral-nemo:12b)
- PDF OCR Support
- Open WebUI Integration
- **Caching System** - Significantly reduced startup time through intelligent caching

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
sudo apt install tesseract-ocr libtesseract-dev
```

2. Start the service:
```bash
./start_rag.sh
```

## Caching System

The RAG pipeline now includes an intelligent caching system that dramatically reduces startup time after the initial run.

### How It Works

1. **Document Processing Cache**: Processed documents are cached with keys based on file paths and modification times
2. **Text Chunking Cache**: Document chunks are cached to avoid re-chunking on subsequent runs
3. **RAPTOR Clustering Cache**: Hierarchical document clusters are cached to avoid recomputation
4. **Index Cache**: Vector indexes are cached where possible (Chroma indexes are rebuilt but BM25 index is cached)
5. **Query Response Cache**: Complete query responses are cached to avoid reprocessing identical queries
6. **Retrieval Cache**: Document retrieval results are cached to avoid recomputing retrieval for identical queries

### Cache Invalidation

Caches are automatically invalidated when:
- Document files are modified (based on file modification times)
- Document files are added or removed
- Cache files are corrupted

### Cache Management Endpoints

The API includes several endpoints for cache management:

- `GET /cache/status` - Get cache status and information
- `POST /cache/clear` - Clear all cached data
- `GET /health` - Check system health including cache status

### Performance Improvement

With caching enabled, startup time is reduced from approximately 2 minutes to just a few seconds on subsequent runs.
Query response time is also significantly improved for repeated queries, as complete responses are cached.

## API Endpoints

- `GET /` - API information
- `POST /query` - Query the RAG pipeline
- `GET /health` - Health check
- `GET /cache/status` - Cache status
- `POST /cache/clear` - Clear cache

## Environment Variables

- `DOCS_PATH` - Path to documents directory (default: /app/documents)
- `DOCS_LANG` - Document language (default: en)
- `OLLAMA_BASE_URL` - Ollama service URL (default: http://ollama:11434)
- `GENERATOR_MODEL` - Response generation model (default: llama3:8b)
- `RERANKER_MODEL` - Re-ranking model (default: mxbai-rerank-large)
- `QUERY_TRANSFORMER_MODEL` - Query transformation model (default: llama3:8b)
- `TRANSLATOR_MODEL` - Translation model (default: mistral-nemo:12b)
- `EMBEDDING_MODEL` - Embedding model (default: nomic-embed-text)