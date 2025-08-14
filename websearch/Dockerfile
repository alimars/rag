# Base image with Python on supported Ubuntu 22.04
FROM python:3.11-slim-bookworm

# Avoid interactive prompts
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for Playwright
RUN apt-get update && \
    apt-get install -y \
    curl \
    gnupg \
    unzip \
    wget \
    build-essential \
    libglib2.0-0 \
    libnss3 \
    libatk-bridge2.0-0 \
    libx11-xcb1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libatk1.0-0 \
    libpangocairo-1.0-0 \
    libcups2 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Install Playwright with dependencies
RUN PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 playwright install --with-deps

# Copy application files
COPY . .

# Expose FastAPI port
EXPOSE 8081

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8081"]
