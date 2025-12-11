# GRPO Trainer Docker Image
# =========================

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

LABEL maintainer="kossisoroyce"
LABEL description="GRPO Trainer - Advanced GRPO Training Framework for LLM Fine-tuning"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    vim \
    build-essential \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Install the package
RUN pip install -e ".[all]"

# Install flash-attn separately (requires special handling)
RUN pip install flash-attn --no-build-isolation || echo "Flash attention installation skipped"

# Create directories for outputs and data
RUN mkdir -p /app/outputs /app/data

# Set default command
CMD ["grpo-train", "--help"]

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import grpo_trainer; print('OK')" || exit 1
