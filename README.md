# Testing-rig

This repository provides a lightweight Coptic OCR demonstration stack.  It
contains a heuristic inference pipeline, a FastAPI service, containerisation
artifacts and a command line interface for batch predictions.

## Quick start

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run a single prediction

Invoke the pipeline using the CLI module:

```bash
python -m src.inference.predict --image path/to/image.png
```

Multiple images can be processed simultaneously and results can be saved to a
file via the `--output` flag.  The command prints rich logging that highlights
progress through preprocessing, OCR and translation stages.

### Start the FastAPI service

```bash
uvicorn src.server.app:app --reload
```

The service exposes the following endpoints:

* `GET /healthz` – liveness probe for the container runtime.
* `GET /readyz` – readiness probe that validates model loading.
* `POST /predict` – accepts a multipart image upload and returns the recognised
  Coptic lines with translations and confidences.

### Docker deployment

Build and run the GPU-enabled container locally using docker compose:

```bash
docker compose up --build
```

By default checkpoints are mounted from `./checkpoints` (read-only).  The Docker
image includes a health check that pings `/healthz` to verify the service is
running.
