FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces requires running as a non-root user (UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirements
COPY --chown=user requirements.txt .

# Install PyTorch 2.11 with cu126 (matching our local battle-tested environment)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 --upgrade

# Install remaining requirements and Triton explicitly for Linux cloud environment
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir triton

# Copy application files
COPY --chown=user . .

# Set PYTHONPATH so Uvicorn can resolve the 'src' module
ENV PYTHONPATH="/home/user/app"

# Expose application port
EXPOSE 8000

# Launch FastAPI app with Uvicorn
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
