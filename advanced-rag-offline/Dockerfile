FROM python:3.11-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    libleptonica-dev \
    g++ \
    make \
    cmake \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install language packs
RUN apt-get update && apt-get install -y \
    tesseract-ocr-eng \
    tesseract-ocr-spa \
    tesseract-ocr-fra \
    tesseract-ocr-deu \
    tesseract-ocr-jpn \
    tesseract-ocr-kor \
    tesseract-ocr-ara \
    tesseract-ocr-chi-sim \
    tesseract-ocr-rus \
    && rm -rf /var/lib/apt/lists/*

# Set TESSDATA_PREFIX
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata/

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Install additional PDF dependencies for unstructured
RUN pip install --no-cache-dir pdfminer.six pdfplumber

# Install FAISS CPU
RUN pip install faiss-cpu

# Install additional dependencies for better error handling
RUN pip install requests

# Copy application
COPY . .

# Make package installable
RUN pip install -e .

# Ensure the rag_tool package is properly installed
RUN python -c "import rag_tool; print('RAG tool package imported successfully')"

# Create documents directory
RUN mkdir -p /app/documents
# Create cache directory
RUN mkdir -p /app/cache

# Environment variables
ENV DOCS_PATH=/app/documents
ENV DOCS_LANG=en
ENV PORT=8000
ENV GENERATOR_MODEL=llama3:8b
ENV RERANKER_MODEL=mxbai-rerank-large
ENV QUERY_TRANSFORMER_MODEL=llama3:8b
ENV TRANSLATOR_MODEL=mistral-nemo:12b
ENV EMBEDDING_MODEL=nomic-embed-text

EXPOSE 8000

CMD ["./start_rag.sh"]