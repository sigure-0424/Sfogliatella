FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements and install
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir \
    "jax[cuda12]>=0.4.20" \
    "equinox>=0.11.0" \
    "optax>=0.2.0" \
    numpy \
    pandas \
    pyarrow \
    xgboost \
    matplotlib \
    watchdog \
    pyyaml \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Copy project
COPY . /workspace/

# Set PYTHONPATH so `python -m train` works from /workspace
ENV PYTHONPATH=/workspace

CMD ["bash"]
