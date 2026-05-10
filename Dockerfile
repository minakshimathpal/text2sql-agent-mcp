# Use Python 3.11 slim image to match local environment
FROM python:3.11-slim

# Prevent Python from writing pyc files to disk and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
# - curl: to download Ollama
# - tesseract-ocr & libtesseract-dev: for OCR fallback
# - gcc & libpq-dev: needed for PostgreSQL and other python packages
RUN apt-get update && apt-get install -y \
    curl \
    zstd \
    tesseract-ocr \
    libtesseract-dev \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama CLI so it can run inside the container
#RUN curl -fsSL https://ollama.com/install.sh | sh

# Set working directory
WORKDIR /app

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose Web UI port
EXPOSE 8000

# Make the startup script executable and fix line endings
RUN sed -i 's/\r$//' start.sh && chmod +x start.sh

# Run the startup script
CMD ["./start.sh"]
