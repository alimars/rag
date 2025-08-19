# OpenWeb UI Integration Guide for RAG Tool

This document explains how to integrate the RAG (Retrieval-Augmented Generation) tool with OpenWeb UI as a custom tool.

## Overview

The RAG tool provides advanced document retrieval and question-answering capabilities with support for:
- Multi-document search and retrieval
- Multi-language support with translation
- Advanced indexing and clustering techniques
- Caching for improved performance
- Privacy-focused offline processing

## Prerequisites

1. The RAG tool service must be running (default on port 8000)
2. OpenWeb UI must be able to access the RAG tool service
3. Required documents should be placed in the `documents` directory

## Integration Steps

### 1. Start the RAG Tool Service

```bash
# Navigate to the advanced-rag-offline directory
cd advanced-rag-offline

# Start the service
./start_rag.sh
```

Or using Docker:
```bash
# From the root directory
docker-compose up -d rag-api
```

### 2. Configure OpenWeb UI

In OpenWeb UI, add a new custom tool with the following configuration:

- **Name**: RAG Tool
- **URL**: `http://localhost:8000/invoke` (adjust host/port as needed)
- **Method**: POST
- **Description**: Advanced RAG pipeline for document-based question answering

### 3. Using the Tool

Once configured, you can use the tool in OpenWeb UI by:
1. Selecting it from the tools panel
2. Providing a query about your documents
3. Optionally specifying a target language for translation

## API Endpoints

The RAG tool provides several endpoints:

### `/invoke` (OpenWeb UI Integration)
- **Method**: POST
- **Purpose**: OpenWeb UI compatible endpoint
- **Input**: 
  ```json
  {
    "query": "Your question here",
    "target_lang": "es",  // Optional: target language for translation
    "return_original": false  // Optional: return original documents instead of generated answer
  }
  ```
- **Output**:
  ```json
  {
    "response": "Generated answer based on documents",
    "translation": "Translated answer (if requested)",
    "source_language": "en",
    "target_language": "es"
  }
  ```

### `/query` (Original API)
- **Method**: POST
- **Purpose**: Original query endpoint
- **Input**: Same as `/invoke` but with different field names
- **Output**: Same structure as `/invoke`

### `/health`
- **Method**: GET
- **Purpose**: Check service health and initialization status

### `/docs`
- **Method**: GET
- **Purpose**: Interactive API documentation

## Supported Languages

The RAG tool supports multiple languages:
- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Chinese (zh)
- Japanese (ja)
- Korean (ko)
- Russian (ru)
- Arabic (ar)

## Configuration

Environment variables can be set to customize the RAG tool:

- `DOCS_PATH`: Path to documents directory (default: ./documents)
- `DOCS_LANG`: Default language of documents (default: en)
- `OLLAMA_BASE_URL`: Ollama service URL (default: http://host.docker.internal:11434)
- `GENERATOR_MODEL`: Model for answer generation (default: llama3:8b)
- `RERANKER_MODEL`: Model for result reranking (default: mxbai-rerank-large)
- `QUERY_TRANSFORMER_MODEL`: Model for query transformation (default: llama3:8b)
- `TRANSLATOR_MODEL`: Model for translation (default: mistral-nemo:12b)
- `EMBEDDING_MODEL`: Model for text embeddings (default: nomic-embed-text)

## Troubleshooting

### Common Issues

1. **Service not responding**: Ensure the RAG tool service is running and accessible
2. **Initialization errors**: Check that documents are properly formatted and accessible
3. **Language detection issues**: Ensure documents are in a supported language
4. **Translation failures**: Verify the translator model is available in Ollama

### Logs

Check the service logs for detailed error information:
```bash
docker-compose logs rag-api
```

## Development

### Package Structure

The RAG tool is structured as a Python package with the following modules:
- `rag_tool.pipeline`: Main RAG pipeline implementation
- `rag_tool.document_processor`: Document loading and processing
- `rag_tool.indexing`: Index construction and search
- `rag_tool.retrieval`: Document retrieval system
- `rag_tool.translation`: Translation capabilities
- `rag_tool.query_transformer`: Query enhancement techniques

### Direct Import

The tool can be imported and used directly in Python:
```python
from rag_tool import FocusedRAGPipeline

pipeline = FocusedRAGPipeline("./documents", "en")
pipeline.initialize()
result = pipeline.query("What is the capital of France?")
```

## Performance Tips

1. **Caching**: The tool uses caching to improve response times for repeated queries
2. **Initialization**: The first run will be slower due to index building
3. **Document Format**: PDF and DOCX files are supported; OCR is used for scanned documents
4. **Batch Processing**: For multiple queries, keep the service running to benefit from caching

## Security

- All processing is done offline for privacy
- No data is sent to external services
- Document access is restricted to the specified documents directory