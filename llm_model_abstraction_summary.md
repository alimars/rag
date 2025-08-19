# LLM Model Abstraction Summary

## Overview
This document summarizes the changes made to abstract hardcoded LLM models from the codebase to environment variables in the docker-compose file.

## Changes Made

### 1. Docker Compose Configuration
Updated `docker-compose.yml` to include environment variables for all LLM models:
- `GENERATOR_MODEL=llama3:8b`
- `RERANKER_MODEL=mxbai-rerank-large`
- `QUERY_TRANSFORMER_MODEL=llama3:8b`
- `TRANSLATOR_MODEL=mistral-nemo:12b`
- `EMBEDDING_MODEL=nomic-embed-text`

### 2. Application Code Changes

#### pipeline.py
- Replaced hardcoded `"llama3:8b"` with `os.getenv("GENERATOR_MODEL", "llama3:8b")`

#### retrieval.py
- Replaced hardcoded `"mxbai-rerank-large"` with `os.getenv("RERANKER_MODEL", "mxbai-rerank-large")`

#### query_transformer.py
- Replaced hardcoded `"llama3:8b"` with `os.getenv("QUERY_TRANSFORMER_MODEL", "llama3:8b")`

#### translation.py
- Replaced hardcoded `"mistral-nemo:12b"` with `os.getenv("TRANSLATOR_MODEL", "mistral-nemo:12b")`
- Replaced hardcoded `"nomic-embed-text"` with `os.getenv("EMBEDDING_MODEL", "nomic-embed-text")`

#### indexing.py
- Replaced hardcoded `"nomic-embed-text"` with `os.getenv("EMBEDDING_MODEL", "nomic-embed-text")` (used in two places)

### 3. Shell Script Updates

#### start_rag.sh
- Added default environment variables for all models
- Updated model pulling section to use environment variables
- Updated echo statements to show the actual models being used
- Modified uvicorn command to pass all environment variables to the application

### 4. Dockerfile Updates
- Added default environment variables for all models to the Dockerfile

## Benefits

1. **Flexibility**: Users can now easily change LLM models without modifying code
2. **Configuration Management**: All model configuration is now centralized in the docker-compose file
3. **Environment-Specific Configuration**: Different environments (dev, test, prod) can use different models
4. **Easier Testing**: Testing with different models is now much simpler
5. **Better Documentation**: The docker-compose file now clearly shows all model dependencies

## Usage

### Default Configuration
The application will use the default models specified in the docker-compose file.

### Custom Configuration
To use custom models, simply modify the environment variables in the docker-compose.yml file:

```yaml
environment:
  - OLLAMA_BASE_URL=http://host.docker.internal:11434
  - GENERATOR_MODEL=llama3:70b
  - RERANKER_MODEL=mxbai-rerank-large
  - QUERY_TRANSFORMER_MODEL=llama3:70b
  - TRANSLATOR_MODEL=mistral-nemo:12b
  - EMBEDDING_MODEL=nomic-embed-text
```

### Command Line Usage
The start_rag.sh script also supports environment variables:

```bash
GENERATOR_MODEL=llama3:70b RERANKER_MODEL=mxbai-rerank-large ./start_rag.sh
```

## Testing
See `llm_model_abstraction_test_plan.md` for detailed testing instructions.