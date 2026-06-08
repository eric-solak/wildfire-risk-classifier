FROM python:3.11-slim

WORKDIR /app

# Install PyTorch CPU-only first (avoids pulling the 2GB+ CUDA build)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.py main.py ./
COPY artifacts/ artifacts/

# Render injects $PORT; default to 8000 locally
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
