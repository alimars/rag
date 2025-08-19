# RAG Tool Integration Plan for OpenWeb UI

## Current State Analysis

### RAG Tool Response Format
The current RAG tool returns responses in this format:
```python
{
    "original_response": response,
    "translation": translation,
    "source_language": query_language
}
```

### OpenWeb UI Expected Format
Based on the websearch example, OpenWeb UI expects tools to:
1. Have an `/invoke` endpoint that accepts a specific input format
2. Return a specific output format
3. Include OpenAPI metadata with `x-openwebui` configuration

## Required Changes

### 1. Make RAG Tool Importable
- Create proper `__init__.py` file in the rag_tool package
- Add necessary imports and exports

### 2. Update Response Format
- Modify the response format to match OpenWeb UI expectations
- Create a new endpoint `/invoke` that follows the ToolInput/ToolOutput pattern

### 3. Add OpenAPI Metadata
- Add `x-openwebui` metadata to mark the tool as a RAG tool
- Configure appropriate icon and default settings

## Implementation Steps

1. Create `__init__.py` file with proper module initialization
2. Add a new function that can be directly called by OpenWeb UI
3. Modify the web API to include the `/invoke` endpoint
4. Add OpenAPI metadata for OpenWeb UI integration
5. Test the integration

## Expected Tool Interface

The final tool should be usable in OpenWeb UI as a custom tool with the following interface:
- Input: Query text and optional parameters
- Output: Structured response with answer and source information
- Metadata: Properly configured for OpenWeb UI integration