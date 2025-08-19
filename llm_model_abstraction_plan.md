# LLM Model Abstraction Plan

## Overview
This plan outlines the steps to abstract hardcoded LLM models from the codebase to environment variables in the docker-compose file.

## Models to Abstract

1. **GENERATOR_MODEL** - "llama3:8b" in `pipeline.py`
2. **RERANKER_MODEL** - "mxbai-rerank-large" in `retrieval.py`
3. **QUERY_TRANSFORMER_MODEL** - "llama3:8b" in `query_transformer.py`
4. **TRANSLATOR_MODEL** - "mistral-nemo:12b" in `translation.py`
5. **EMBEDDING_MODEL** - "nomic-embed-text" in `translation.py` and `indexing.py`

## Implementation Steps

1. Update `docker-compose.yml` to include the new environment variables with default values
2. Modify `pipeline.py` to use `GENERATOR_MODEL` environment variable
3. Modify `retrieval.py` to use `RERANKER_MODEL` environment variable
4. Modify `query_transformer.py` to use `QUERY_TRANSFORMER_MODEL` environment variable
5. Modify `translation.py` to use `TRANSLATOR_MODEL` and `EMBEDDING_MODEL` environment variables
6. Modify `indexing.py` to use `EMBEDDING_MODEL` environment variable
7. Update `start_rag.sh` to include the new environment variables
8. Test the changes to ensure everything works correctly

## Docker Compose Environment Variables

```yaml
environment:
  - OLLAMA_BASE_URL=http://host.docker.internal:11434
  - GENERATOR_MODEL=llama3:8b
  - RERANKER_MODEL=mxbai-rerank-large
  - QUERY_TRANSFORMER_MODEL=llama3:8b
  - TRANSLATOR_MODEL=mistral-nemo:12b
  - EMBEDDING_MODEL=nomic-embed-text
```

## File Modifications

### pipeline.py
Change line 17 from:
```python
self.generator = OllamaLLM(model="llama3:8b", base_url=ollama_base_url)
```
to:
```python
generator_model = os.getenv("GENERATOR_MODEL", "llama3:8b")
self.generator = OllamaLLM(model=generator_model, base_url=ollama_base_url)
```

### retrieval.py
Change line 9 from:
```python
self.reranker = OllamaLLM(base_url=ollama_base_url, model="mxbai-rerank-large", temperature=0)
```
to:
```python
reranker_model = os.getenv("RERANKER_MODEL", "mxbai-rerank-large")
self.reranker = OllamaLLM(base_url=ollama_base_url, model=reranker_model, temperature=0)
```

### query_transformer.py
Change line 9 from:
```python
self.llm = OllamaLLM(model="llama3:8b", base_url=ollama_base_url, temperature=0.3)
```
to:
```python
query_transformer_model = os.getenv("QUERY_TRANSFORMER_MODEL", "llama3:8b")
self.llm = OllamaLLM(model=query_transformer_model, base_url=ollama_base_url, temperature=0.3)
```

### translation.py
Change line 11 from:
```python
self.translator = OllamaLLM(model="mistral-nemo:12b", base_url=ollama_base_url)
```
to:
```python
translator_model = os.getenv("TRANSLATOR_MODEL", "mistral-nemo:12b")
self.translator = OllamaLLM(model=translator_model, base_url=ollama_base_url)
```

Change line 62 from:
```python
model="nomic-embed-text"
```
to:
```python
model=embedding_model
```

### indexing.py
Change line 21 from:
```python
model="nomic-embed-text"
```
to:
```python
model=embedding_model
```

Change line 35 from:
```python
model="nomic-embed-text"
```
to:
```python
model=embedding_model
```

## Testing
After implementing the changes, verify that:
1. The application starts correctly
2. All environment variables have appropriate default values
3. The application functions as expected with the default models
4. Custom models can be specified through environment variables