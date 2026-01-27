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

# REMOVED: Healthcheck instruction in Dockerfile
# Cloud Run has its own built-in health checking mechanism (Startup/Liveness probes).
# Defining HEALTHCHECK here can sometimes conflict or be redundant if not perfectly aligned.
# We rely on Cloud Run's TCP probe to port 8080.

# Expose port (Documentation only, Cloud Run overrides this)
EXPOSE 8080

# Run the application
# Cloud Run sets PORT environment variable; uvicorn reads it in main.py
# Added --host 0.0.0.0 explicitly to ensure we bind to all interfaces
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
