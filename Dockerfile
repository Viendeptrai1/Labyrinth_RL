# Use existing Zeppelin image that works on this machine
FROM zeppelin-spark-project-zeppelin:latest

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages for RL
RUN pip install --no-cache-dir \
    torch>=2.0.0 \
    numpy>=1.24.0 \
    gymnasium>=0.29.0 \
    matplotlib>=3.7.0 \
    tqdm>=4.65.0 \
    orjson>=3.9.0

# Copy source code
COPY src /src

# Add src to Python path
ENV PYTHONPATH="${PYTHONPATH}:/src"

# Keep original WORKDIR for Zeppelin to start correctly
# WORKDIR stays at /opt/zeppelin

USER 1000
