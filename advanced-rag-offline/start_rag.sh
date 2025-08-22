#!/bin/bash

# Set defaults
DEFAULT_DOCS_PATH="/app/documents"
DEFAULT_DOCS_LANG="en"
DEFAULT_PORT=8000
DEFAULT_GENERATOR_MODEL="llama3:8b"
DEFAULT_QUERY_TRANSFORMER_MODEL="llama3:8b"
DEFAULT_TRANSLATOR_MODEL="mistral-nemo:12b"
DEFAULT_EMBEDDING_MODEL="jeffh/intfloat-multilingual-e5-large-instruct:Q8_0"

# Use environment variables if set, otherwise use command-line arguments
DOCS_PATH="${DOCS_PATH:-${1:-$DEFAULT_DOCS_PATH}}"
DOCS_LANG="${DOCS_LANG:-${2:-$DEFAULT_DOCS_LANG}}"
PORT="${PORT:-${3:-$DEFAULT_PORT}}"
GENERATOR_MODEL="${GENERATOR_MODEL:-$DEFAULT_GENERATOR_MODEL}"
QUERY_TRANSFORMER_MODEL="${QUERY_TRANSFORMER_MODEL:-$DEFAULT_QUERY_TRANSFORMER_MODEL}"
TRANSLATOR_MODEL="${TRANSLATOR_MODEL:-$DEFAULT_TRANSLATOR_MODEL}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-$DEFAULT_EMBEDDING_MODEL}"

# Verify Tesseract language
TESS_LANG=$(python -c "from rag_tool.document_processor import LANGUAGE_MAP; print(LANGUAGE_MAP.get('$DOCS_LANG', 'eng'))")
if ! tesseract --list-langs | grep -q "$TESS_LANG"; then
    echo "⚠️ Tesseract language $TESS_LANG not installed. Defaulting to English."
    DOCS_LANG="en"
fi

echo "🚀 Starting RAG API on port $PORT"
echo "📁 Document path: $DOCS_PATH"
echo "🌐 Document language: $DOCS_LANG"
echo "🔒 Privacy: All processing offline"
echo "🔤 Translation: Enabled ($TRANSLATOR_MODEL model)"
echo "🧠 Generator model: $GENERATOR_MODEL"
echo "🔍 Query transformer model: $QUERY_TRANSFORMER_MODEL"
echo "🔗 Embedding model: $EMBEDDING_MODEL"
echo "💾 Caching: Enabled (persistent cache at ./cache)"
echo "⚡ Performance: Subsequent startups will be much faster"
echo "�️ TESSDATA_PREFIX: $TESSDATA_PREFIX"

# Check if we're using a remote Ollama instance
if [[ "$OLLAMA_BASE_URL" == *"host.docker.internal"* ]] || [[ "$OLLAMA_BASE_URL" == *"localhost"* ]] || [[ "$OLLAMA_BASE_URL" == *"127.0.0.1"* ]]; then
    echo "📡 Connecting to remote Ollama instance at $OLLAMA_BASE_URL"
    echo "⚠️  Skipping model pulling and readiness check for remote instance"
    echo "💡 Make sure the following models are available in your local Ollama:"
    echo "   - $EMBEDDING_MODEL"
    echo "   - $TRANSLATOR_MODEL"
    echo "   - $GENERATOR_MODEL"
    echo "   - $QUERY_TRANSFORMER_MODEL"
    
    # Check connectivity to remote Ollama
    echo "🔍 Testing connectivity to remote Ollama..."
    if ! curl -s "$OLLAMA_BASE_URL/api/tags" >/dev/null 2>&1; then
        echo "❌ Cannot connect to Ollama at $OLLAMA_BASE_URL"
        echo "💡 Please ensure Ollama is running on your host machine and accessible"
        exit 1
    fi
    echo "✅ Connected to remote Ollama successfully!"
else
    # Pull required Ollama models
    echo "📥 Pulling required Ollama models..."
    ollama pull $EMBEDDING_MODEL
    ollama pull $TRANSLATOR_MODEL
    ollama pull $GENERATOR_MODEL
    ollama pull $QUERY_TRANSFORMER_MODEL

    # Wait for Ollama to be ready
    echo "⏳ Waiting for Ollama to be ready..."
    timeout=60
    counter=0
    while ! ollama list >/dev/null 2>&1; do
        counter=$((counter + 1))
        if [ $counter -ge $timeout ]; then
            echo "❌ Ollama failed to start within $timeout seconds"
            exit 1
        fi
        echo "⏳ Waiting for Ollama... ($counter/$timeout)"
        sleep 1
    done
    echo "✅ Ollama is ready!"
fi

# Start the API
echo "🚀 Starting the RAG API server..."
uvicorn web_api:app --host 0.0.0.0 --port $PORT --env-file <(env | grep -E 'DOCS_PATH|DOCS_LANG|OLLAMA_BASE_URL|GENERATOR_MODEL|QUERY_TRANSFORMER_MODEL|TRANSLATOR_MODEL|EMBEDDING_MODEL') 2>&1 | tee rag_api.log