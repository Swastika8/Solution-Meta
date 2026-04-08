# Dockerfile

FROM python:3.11-slim

# System optimizations
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install minimal dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Resource constraints hint (handled by runtime, but documented)
# Optimized for: 2 vCPU, 8GB RAM

CMD ["python", "inference.py", "--task", "task_easy"]
