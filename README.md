# Testing-rig

This repository provides a minimal machine-learning pipeline built around a
synthetic regression task. It includes preprocessing helpers, dataset
management, a simple training loop, and inference utilities. Artefacts are
tracked through a lightweight on-disk registry located in `mlruns/`.

## Development setup

Install the dependencies and run the unit test suite:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
make test
```

## Tooling

- `make lint` – run Ruff and Black.
- `make format` – auto-format using Black and Ruff (with autofix).
- `make test` – execute the pytest suite.
- `make train-preview` – run a lightweight training loop that logs artefacts to
  the local registry.

Alternatively, use `tox`:

```bash
tox -e lint
tox -e py
```

## Continuous integration

GitHub Actions is configured to lint and test on every push or pull request.
Nightly scheduled runs execute a training smoke test to validate the registry
integration and end-to-end workflow.
