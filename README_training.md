# Training Pipeline

This repository provides a lightweight PyTorch Lightning training loop for speech-recognition style models. The entry-point is [`src/training/train.py`](src/training/train.py), which consumes configuration from [`configs/train.yaml`](configs/train.yaml) and produces checkpoints under `artifacts/checkpoints/`.

## Features

- Synthetic speech dataset that is handy for integration tests and experimenting with the pipeline structure.
- Configurable model, optimizer (AdamW), learning-rate scheduler, and precision settings via YAML.
- Gradient clipping, validation CER/WER calculation, TensorBoard/W&B logging, early stopping, and checkpoint management out of the box.
- Utilities for computing CER/WER, scheduling helpers, and mixed precision configuration in [`src/training/utils.py`](src/training/utils.py).

## Prerequisites

Install the required dependencies:

```bash
pip install torch torchvision torchaudio lightning tensorboard
# Optional: for experiment tracking
pip install wandb
```

## Running Training

The default configuration trains a compact CTC model on synthetic data generated on the fly.

```bash
python -m src.training.train --config configs/train.yaml
```

Key artifacts will be written to:

- **Checkpoints:** `artifacts/checkpoints/`
- **TensorBoard logs:** `artifacts/logs/tensorboard/`
- **W&B run (optional):** configure `logging.use_wandb: true`

### Fast debugging

Use the `--fast-dev-run` flag to verify the setup without running a full epoch:

```bash
python -m src.training.train --config configs/train.yaml --fast-dev-run
```

To resume from a previous checkpoint:

```bash
python -m src.training.train --config configs/train.yaml \
    --resume-from-checkpoint artifacts/checkpoints/ctc-epoch.ckpt
```

## Customising the pipeline

- **Datasets/DataModule:** Update the `data` section of the YAML config to point at your own dataset implementation. The config expects dotted Python paths.
- **Model:** Provide a `model.target` in the config with constructor parameters that accept `vocab`, `scheduler`, and `blank_id` keywords. Otherwise the default `SimpleCTCModule` is used.
- **Logging:** Enable Weights & Biases by setting `logging.use_wandb: true` and supplying `logging.project` / `logging.run_name`.
- **Mixed precision:** Configure the `precision` stanza to forward options directly to the Lightning `Trainer`.

Check `configs/train.yaml` for a comprehensive example of available knobs.
