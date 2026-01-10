# Multi-stage build for Podcasts TLDR Autonomous Pipeline
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -u 1000 -s /bin/bash podcasts && \
    mkdir -p /app && \
    chown podcasts:podcasts /app

# Set working directory
WORKDIR /app

# Copy requirements first (for layer caching)
COPY --chown=podcasts:podcasts requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=podcasts:podcasts . .

# Create required directories
RUN mkdir -p data logs output cache backups && \
    chown -R podcasts:podcasts /app

# Switch to non-root user
USER podcasts

# Health check
HEALTHCHECK --interval=5m --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import sys; from src.health_monitor import HealthMonitor; \
                   m = HealthMonitor(); status, _ = m.check_all(); \
                   sys.exit(0 if status.value == 'healthy' else 1)"

# Default command: Run daemon (sets up env vars and runs orchestrator)
CMD ["python", "daemon.py"]
