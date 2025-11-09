FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN python3 -m pip install --upgrade pip && \
    pip install -r requirements.txt

COPY src ./src
COPY README.md ./

ENV PYTHONPATH=/app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "src.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
