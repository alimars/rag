# Plan to Fix Application Hang Issue

## Problem Analysis
The application is hanging at the "waiting for application startup" phase. Based on code analysis, the issue is likely occurring during the pipeline initialization process in the startup event handler.

## Root Causes Identified
1. **Ollama Connectivity Issues**: The application tries to connect to Ollama services which might not be available or responsive
2. **Embedding Process Problems**: The embedding process in the indexing module might be hanging due to model unavailability
3. **Document Processing Issues**: Large documents or OCR processing might be taking too long
4. **Missing Timeouts**: No timeout mechanisms exist to prevent indefinite hanging

## Proposed Fixes

### 1. Improve Error Handling in web_api.py
- Add more detailed logging to identify exactly where the hang occurs
- Add try-catch blocks around each initialization step
- Implement timeout mechanisms for Ollama connectivity checks

### 2. Add Ollama Connectivity Check
- Before initializing the pipeline, verify that Ollama is accessible
- Check that required models are available
- Implement a timeout for connectivity checks

### 3. Add Timeout Mechanisms
- Implement timeouts for embedding operations
- Add timeouts for document processing steps
- Use Python's `concurrent.futures` with timeout parameters

### 4. Improve Caching Robustness
- Add better error handling for cache operations
- Implement cache validation to prevent corrupted cache issues
- Add cache cleanup mechanisms for failed operations

### 5. Add Health Check Improvements
- Enhance the health check endpoint to provide more detailed status information
- Add initialization progress tracking

## Implementation Steps

### Step 1: Modify web_api.py
- Enhance the startup event handler with better error handling
- Add detailed logging for each initialization step
- Implement Ollama connectivity check with timeout

### Step 2: Modify pipeline.py
- Add timeout mechanisms to the initialize() method
- Improve error handling for document processing
- Add progress indicators for long-running operations

### Step 3: Modify translation.py
- Add timeout to the embed_text function
- Improve error handling for embedding operations

### Step 4: Modify indexing.py
- Add timeout mechanisms for Chroma index creation
- Improve error handling for index building

### Step 5: Modify document_processor.py
- Add timeout mechanisms for document loading
- Improve error handling for OCR operations

## Files to Modify
1. `advanced-rag-offline/web_api.py` - Main startup handler
2. `advanced-rag-offline/rag_tool/pipeline.py` - Pipeline initialization
3. `advanced-rag-offline/rag_tool/translation.py` - Embedding operations
4. `advanced-rag-offline/rag_tool/indexing.py` - Index building
5. `advanced-rag-offline/rag_tool/document_processor.py` - Document loading

## Testing Plan
1. Test with Ollama service running and models available
2. Test with Ollama service not running
3. Test with missing models
4. Test with large document sets
5. Test cache behavior with various scenarios

## Expected Outcomes
- Application will no longer hang indefinitely during startup
- Clear error messages will be provided when issues occur
- Timeout mechanisms will prevent long waits
- Better logging will help diagnose future issues