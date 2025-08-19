# LLM Model Abstraction Test Plan

## Overview
This test plan outlines the steps to verify that the LLM model abstraction changes work correctly.

## Test Cases

### 1. Default Environment Variables
- Start the application using the default docker-compose configuration
- Verify that all models are loaded correctly with their default values
- Check that the application functions as expected

### 2. Custom Environment Variables
- Modify the docker-compose.yml file to use custom model values
- Start the application
- Verify that the custom models are loaded instead of the defaults
- Check that the application functions correctly with the custom models

### 3. Partial Custom Environment Variables
- Modify the docker-compose.yml file to override only some of the environment variables
- Start the application
- Verify that the overridden variables use the custom values while the others use defaults
- Check that the application functions correctly

### 4. Command Line Arguments
- Start the application using the start_rag.sh script with command line arguments
- Verify that the models are loaded correctly
- Check that the application functions as expected

### 5. Mixed Configuration
- Use a combination of environment variables and command line arguments
- Verify that the correct precedence is applied
- Check that the application functions correctly

## Test Commands

### Default Configuration
```bash
docker-compose up
```

### Custom Configuration
```yaml
environment:
  - OLLAMA_BASE_URL=http://host.docker.internal:11434
  - GENERATOR_MODEL=llama3:70b
  - RERANKER_MODEL=mxbai-rerank-large
  - QUERY_TRANSFORMER_MODEL=llama3:70b
  - TRANSLATOR_MODEL=mistral-nemo:12b
  - EMBEDDING_MODEL=nomic-embed-text
```

### Command Line Arguments
```bash
./start_rag.sh /app/documents en 8000
```

## Expected Results

1. The application should start without errors
2. All models should be loaded correctly based on the configuration
3. The application should function as expected in all test scenarios
4. Environment variables should be properly passed to the application
5. Default values should be used when environment variables are not specified

## Verification Steps

1. Check the application logs for model loading messages
2. Verify that the correct models are being used in the logs
3. Test the application functionality with sample queries
4. Verify that translations work correctly
5. Check that document processing and retrieval work as expected