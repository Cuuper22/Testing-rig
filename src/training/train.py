"""Entry point for launching distributed training with PyTorch Lightning or
Hugging Face Accelerate.

The script consumes a YAML configuration file (see ``configs/distributed.yaml``)
that specifies which backend to use and how to configure it. An extremely small
random dataset is used to demonstrate the distributed setup without requiring a
real dataset.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

try:
    import pytorch_lightning as pl
except ImportError:  # pragma: no cover - informative error for missing optional dependency
    pl = None

try:
    from accelerate import Accelerator
except ImportError:  # pragma: no cover
    Accelerator = None


class RandomClassificationDataset(Dataset):
    """Toy dataset that yields random features and labels.

    The dataset is intentionally tiny and synthetic so the script can run
    without relying on external data sources. It is only meant to demonstrate
    the distributed training hooks.
    """

    def __init__(self, length: int, input_dim: int, num_classes: int) -> None:
        self.length = length
        self.input_dim = input_dim
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):  # type: ignore[override]
        features = torch.randn(self.input_dim)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return features, label


class ExampleModel(nn.Module):
    """Tiny multi-layer perceptron for demonstration purposes."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.network(x)


class LightningTrainingModule(pl.LightningModule):  # type: ignore[misc]
    """Wraps :class:`ExampleModel` for use with PyTorch Lightning."""

    def __init__(self, model_cfg: Dict[str, Any]) -> None:
        if pl is None:  # pragma: no cover - guard in case Lightning is missing
            raise RuntimeError(
                "PyTorch Lightning is not installed. Install it or switch the "
                "backend to 'accelerate'."
            )
        super().__init__()
        self.save_hyperparameters(model_cfg)
        self.model = ExampleModel(
            model_cfg["input_dim"],
            model_cfg["hidden_dim"],
            model_cfg["num_classes"],
        )
        self.learning_rate = model_cfg.get("lr", 1e-3)
        self.batch_size = model_cfg.get("batch_size", 32)
        self.dataset_length = model_cfg.get("dataset_length", 1024)

    def forward(self, features: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.model(features)

    def training_step(self, batch, batch_idx: int):  # type: ignore[override]
        features, labels = batch
        logits = self(features)
        loss = F.cross_entropy(logits, labels)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):  # type: ignore[override]
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def train_dataloader(self):  # type: ignore[override]
        dataset = RandomClassificationDataset(
            length=self.dataset_length,
            input_dim=self.hparams.input_dim,
            num_classes=self.hparams.num_classes,
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )


def run_lightning(config: Dict[str, Any], args: argparse.Namespace) -> None:
    if pl is None:
        raise RuntimeError(
            "PyTorch Lightning is not installed. Install it to use the 'lightning' backend."
        )

    trainer_cfg = config.get("trainer", {})
    model_cfg = config.get("model", {})

    module = LightningTrainingModule(model_cfg)

    trainer_kwargs: Dict[str, Any] = {
        "accelerator": trainer_cfg.get("accelerator", "auto"),
        "devices": trainer_cfg.get("devices", "auto"),
        "num_nodes": trainer_cfg.get("num_nodes", 1),
        "strategy": trainer_cfg.get("strategy", "auto"),
        "precision": trainer_cfg.get("precision", 32),
        "accumulate_grad_batches": trainer_cfg.get("accumulate_grad_batches", 1),
        "max_epochs": trainer_cfg.get("max_epochs", 1),
        "log_every_n_steps": trainer_cfg.get("log_every_n_steps", 50),
        "enable_checkpointing": trainer_cfg.get("enable_checkpointing", True),
        "deterministic": trainer_cfg.get("deterministic", False),
        "default_root_dir": trainer_cfg.get("default_root_dir", "runs"),
        "limit_train_batches": trainer_cfg.get("limit_train_batches", 10),
    }

    if trainer_cfg.get("tpu_cores") is not None:
        trainer_kwargs["tpu_cores"] = trainer_cfg["tpu_cores"]

    logging.info(
        "Starting PyTorch Lightning training on node %s/%s",
        args.node_rank,
        args.world_size,
    )
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(module)


def run_accelerate(config: Dict[str, Any], args: argparse.Namespace) -> None:
    if Accelerator is None:
        raise RuntimeError(
            "Hugging Face Accelerate is not installed. Install it to use the 'accelerate' backend."
        )

    model_cfg = config.get("model", {})
    accelerate_cfg = config.get("accelerate", {})

    accelerator = Accelerator(
        mixed_precision=accelerate_cfg.get("mixed_precision", "no"),
        gradient_accumulation_steps=accelerate_cfg.get("gradient_accumulation_steps", 1),
        log_with=accelerate_cfg.get("log_with"),
        project_dir=accelerate_cfg.get("logging_dir", "runs"),
    )

    model = ExampleModel(
        model_cfg.get("input_dim", 32),
        model_cfg.get("hidden_dim", 64),
        model_cfg.get("num_classes", 10),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=model_cfg.get("lr", 1e-3))

    dataset = RandomClassificationDataset(
        length=model_cfg.get("dataset_length", 1024),
        input_dim=model_cfg.get("input_dim", 32),
        num_classes=model_cfg.get("num_classes", 10),
    )
    dataloader = DataLoader(dataset, batch_size=model_cfg.get("batch_size", 32), shuffle=True)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    num_epochs = accelerate_cfg.get("num_epochs", 1)
    steps_per_epoch = accelerate_cfg.get("steps_per_epoch", len(dataloader))

    logging.info(
        "Starting Accelerate training on process %s/%s",
        args.node_rank,
        args.world_size,
    )

    for epoch in range(num_epochs):
        for step, (features, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(features)
            loss = F.cross_entropy(outputs, labels)
            accelerator.backward(loss)
            optimizer.step()

            if step + 1 >= steps_per_epoch:
                break

        accelerator.print(f"Epoch {epoch + 1}/{num_epochs} - loss: {loss.item():.4f}")

    accelerator.wait_for_everyone()
    output_dir = Path(accelerate_cfg.get("output_dir", "checkpoints"))
    output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.save_state(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch distributed training")
    parser.add_argument("--config", type=Path, default=Path("configs/distributed.yaml"))
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=str, default="29500")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    logging_cfg = config.get("logging", {})
    configure_logging(logging_cfg.get("level", "INFO"))

    backend = config.get("backend", "lightning").lower()

    if backend == "lightning":
        run_lightning(config, args)
    elif backend == "accelerate":
        run_accelerate(config, args)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


if __name__ == "__main__":
    main()
