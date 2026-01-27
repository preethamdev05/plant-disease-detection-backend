# Multi-stage Dockerfile for Cloud Run
# Optimized for minimal size and secure production deployment

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py .
COPY inference_metadata.json .

# Health check (optional but recommended)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8080/health')"

# Expose port
EXPOSE 8080

# Run the application
# Cloud Run sets PORT environment variable; uvicorn reads it in main.py
CMD ["python", "-u", "main.py"]
