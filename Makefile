.PHONY: test lint format train-preview

PYTHON=python

lint:
$(PYTHON) -m ruff check .
$(PYTHON) -m black --check .

format:
$(PYTHON) -m black .
$(PYTHON) -m ruff check --fix .

test:
$(PYTHON) -m pytest

train-preview:
$(PYTHON) scripts/train_preview.py
