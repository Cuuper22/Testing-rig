"""Training entry-point for speech recognition models."""
from __future__ import annotations

import argparse
import importlib
import logging
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import LightningLoggerBase, TensorBoardLogger
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from yaml import safe_load

from .utils import calculate_cer, calculate_wer, get_precision_kwargs, get_scheduler

LOGGER = logging.getLogger(__name__)


class DummySpeechDataset(Dataset):
    """A light-weight dataset that generates random audio features and labels."""

    def __init__(
        self,
        size: int = 128,
        input_features: int = 80,
        max_time_steps: int = 160,
        min_time_steps: int = 80,
        min_label_length: int = 4,
        max_label_length: int = 18,
        vocab: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.size = size
        self.input_features = input_features
        self.max_time_steps = max_time_steps
        self.min_time_steps = min_time_steps
        self.min_label_length = min_label_length
        self.max_label_length = max_label_length
        self.rng = random.Random(seed)
        vocab = vocab or list("abcdefghijklmnopqrstuvwxyz '")
        if "<blank>" in vocab:
            self.tokens = list(vocab)
        else:
            self.tokens = ["<blank>"] + list(vocab)
        self.blank_id = self.tokens.index("<blank>")

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str | int]:
        del index
        time_steps = self.rng.randint(self.min_time_steps, self.max_time_steps)
        label_length = self.rng.randint(self.min_label_length, self.max_label_length)

        inputs = torch.randn(time_steps, self.input_features, dtype=torch.float32)
        label_ids = torch.tensor(
            [self.rng.randint(1, len(self.tokens) - 1) for _ in range(label_length)],
            dtype=torch.long,
        )
        transcript = "".join(self.tokens[idx] for idx in label_ids.tolist())
        return {
            "inputs": inputs,
            "labels": label_ids,
            "transcript": transcript,
            "input_length": time_steps,
        }


class SpeechCollator:
    """Pad variable-length examples into a batch that is compatible with CTC."""

    def __init__(self, blank_id: int = 0) -> None:
        self.blank_id = blank_id

    def __call__(self, batch: Iterable[Dict[str, torch.Tensor | str | int]]) -> Dict[str, torch.Tensor | List[str]]:
        inputs = [item["inputs"] for item in batch]
        input_lengths = torch.tensor([item["inputs"].shape[0] for item in batch], dtype=torch.long)
        labels = [item["labels"] for item in batch]
        label_lengths = torch.tensor([len(item) for item in labels], dtype=torch.long)
        transcripts = [str(item["transcript"]) for item in batch]

        padded_inputs = pad_sequence(inputs, batch_first=True)
        concatenated_labels = torch.cat(labels, dim=0)

        return {
            "inputs": padded_inputs,
            "input_lengths": input_lengths,
            "labels": concatenated_labels,
            "label_lengths": label_lengths,
            "transcripts": transcripts,
        }


class RandomSpeechDataModule(L.LightningDataModule):
    """Lightning data module that wraps :class:`DummySpeechDataset`."""

    def __init__(
        self,
        train_dataset_config: Optional[Dict[str, object]] = None,
        val_dataset_config: Optional[Dict[str, object]] = None,
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.train_dataset_config = train_dataset_config or {}
        self.val_dataset_config = val_dataset_config or {}
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None):
            self.train_dataset = self._instantiate_dataset(self.train_dataset_config)
            self.val_dataset = self._instantiate_dataset(self.val_dataset_config)

    def _instantiate_dataset(self, config: Dict[str, object]) -> Dataset:
        if not config:
            return DummySpeechDataset()
        target = config.get("target")
        params = config.get("params") or {}
        if not target:
            return DummySpeechDataset(**params)
        module_name, _, class_name = target.rpartition(".")
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls(**params)

    @property
    def blank_id(self) -> int:
        dataset = self.train_dataset or self._instantiate_dataset(self.train_dataset_config)
        if hasattr(dataset, "blank_id"):
            return int(dataset.blank_id)  # type: ignore[no-any-return]
        return 0

    @property
    def vocab(self) -> List[str]:
        dataset = self.train_dataset or self._instantiate_dataset(self.train_dataset_config)
        if hasattr(dataset, "tokens"):
            return list(dataset.tokens)  # type: ignore[attr-defined]
        return ["<blank>"] + list("abcdefghijklmnopqrstuvwxyz '")

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=SpeechCollator(blank_id=self.blank_id),
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=SpeechCollator(blank_id=self.blank_id),
        )


class SimpleCTCModule(L.LightningModule):
    """A small CTC-based model suitable for experiments and CI validation."""

    def __init__(
        self,
        input_features: int,
        vocab: Sequence[str],
        hidden_dim: int = 256,
        dropout: float = 0.1,
        lr: float = 5e-4,
        weight_decay: float = 1e-2,
        scheduler: Optional[Dict[str, object]] = None,
        blank_id: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.vocab = list(vocab)
        self.blank_id = blank_id
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_config = scheduler

        output_dim = len(self.vocab)
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_features),
            nn.Linear(input_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.ctc_loss = nn.CTCLoss(blank=self.blank_id, zero_infinity=True)
        self.total_training_steps: Optional[int] = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            if self.trainer is not None:
                self.total_training_steps = self.trainer.estimated_stepping_batches

    def _shared_step(self, batch: Dict[str, torch.Tensor | List[str]], stage: str) -> torch.Tensor:
        inputs = batch["inputs"].to(self.device)
        input_lengths = batch["input_lengths"].to(self.device)
        labels = batch["labels"].to(self.device)
        label_lengths = batch["label_lengths"].to(self.device)

        logits = self(inputs)
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
        loss = self.ctc_loss(log_probs, labels, input_lengths, label_lengths)

        self.log(f"{stage}_loss", loss, prog_bar=stage == "train", on_step=stage == "train", on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch: Dict[str, torch.Tensor | List[str]], batch_idx: int) -> torch.Tensor:
        del batch_idx
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Dict[str, torch.Tensor | List[str]], batch_idx: int) -> torch.Tensor:
        del batch_idx
        loss = self._shared_step(batch, "val")

        with torch.no_grad():
            predictions = self._greedy_decode(batch)
            references = batch["transcripts"]
            cer = calculate_cer(predictions, references)
            wer = calculate_wer(predictions, references)
            self.log("val_cer", cer, prog_bar=True, on_epoch=True, sync_dist=True)
            self.log("val_wer", wer, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        total_steps = self.total_training_steps
        if total_steps is None and self.trainer is not None:
            total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
        scheduler = get_scheduler(optimizer, self.scheduler_config, total_steps)
        if scheduler is None:
            return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _greedy_decode(self, batch: Dict[str, torch.Tensor | List[str]]) -> List[str]:
        inputs = batch["inputs"].to(self.device)
        input_lengths: torch.Tensor = batch["input_lengths"].to(self.device)
        logits = self(inputs)
        log_probs = logits.log_softmax(dim=-1)
        predictions = log_probs.argmax(dim=-1)

        decoded: List[str] = []
        for idx, sequence in enumerate(predictions):
            length = int(input_lengths[idx].item())
            prev = self.blank_id
            tokens: List[str] = []
            for token_id in sequence[:length].tolist():
                if token_id == self.blank_id:
                    prev = token_id
                    continue
                if token_id == prev:
                    continue
                if 0 <= token_id < len(self.vocab):
                    tokens.append(self.vocab[token_id])
                prev = token_id
            decoded.append("".join(tokens))
        return decoded


def build_callbacks(config: Dict[str, object], checkpoint_dir: Path) -> List[L.Callback]:
    callbacks: List[L.Callback] = []

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_cfg = config.get("checkpoint") or {}
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=checkpoint_cfg.get("filename", "ctc-{epoch:02d}-{val_cer:.4f}"),
        monitor=checkpoint_cfg.get("monitor", "val_cer"),
        mode=checkpoint_cfg.get("mode", "min"),
        save_top_k=int(checkpoint_cfg.get("save_top_k", 3)),
        every_n_epochs=checkpoint_cfg.get("every_n_epochs"),
        save_last=checkpoint_cfg.get("save_last", True),
    )
    callbacks.append(checkpoint_callback)

    early_stopping_cfg = config.get("early_stopping") or {}
    patience = early_stopping_cfg.get("patience")
    if patience is not None:
        callbacks.append(
            EarlyStopping(
                monitor=early_stopping_cfg.get("monitor", "val_cer"),
                patience=int(patience),
                mode=early_stopping_cfg.get("mode", "min"),
                min_delta=float(early_stopping_cfg.get("min_delta", 0.0)),
            )
        )

    callbacks.append(LearningRateMonitor(logging_interval="step"))
    return callbacks


def build_loggers(logging_config: Dict[str, object]) -> List[LightningLoggerBase]:
    log_dir = Path(logging_config.get("log_dir", "artifacts/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_logger = TensorBoardLogger(save_dir=str(log_dir), name="tensorboard")
    loggers: List[LightningLoggerBase] = [tensorboard_logger]

    use_wandb = logging_config.get("use_wandb", False)
    if use_wandb:
        try:
            from lightning.pytorch.loggers import WandbLogger

            wandb_logger = WandbLogger(
                project=logging_config.get("project", "speech-training"),
                name=logging_config.get("run_name"),
                save_dir=str(log_dir),
                log_model=logging_config.get("log_model", False),
            )
            loggers.append(wandb_logger)
        except ImportError:  # pragma: no cover - wandb optional dependency
            LOGGER.warning("W&B is not installed; falling back to TensorBoard logging only.")
    return loggers


def instantiate_datamodule(config: Dict[str, object]) -> RandomSpeechDataModule:
    data_cfg = config.get("data") or {}
    datamodule = RandomSpeechDataModule(
        train_dataset_config=data_cfg.get("train_dataset"),
        val_dataset_config=data_cfg.get("val_dataset"),
        batch_size=int(data_cfg.get("batch_size", 8)),
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", False)),
    )
    return datamodule


def instantiate_model(config: Dict[str, object], datamodule: RandomSpeechDataModule) -> L.LightningModule:
    model_cfg = config.get("model") or {}
    scheduler_cfg = config.get("scheduler")

    dataset = datamodule.train_dataset or datamodule._instantiate_dataset(datamodule.train_dataset_config)
    if hasattr(dataset, "input_features"):
        input_features = int(dataset.input_features)  # type: ignore[attr-defined]
    else:
        input_features = int(model_cfg.get("input_features", 80))

    vocab = datamodule.vocab
    model_target = model_cfg.get("target")
    model_params = model_cfg.get("params") or {}

    if model_target:
        module_name, _, class_name = model_target.rpartition(".")
        module = importlib.import_module(module_name)
        model_cls = getattr(module, class_name)
        model = model_cls(vocab=vocab, scheduler=scheduler_cfg, blank_id=datamodule.blank_id, **model_params)
    else:
        model = SimpleCTCModule(
            input_features=input_features,
            vocab=vocab,
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            lr=float((config.get("optimizer") or {}).get("lr", 5e-4)),
            weight_decay=float((config.get("optimizer") or {}).get("weight_decay", 1e-2)),
            scheduler=scheduler_cfg,
            blank_id=datamodule.blank_id,
        )

    return model


def setup_optimizer_params(config: Dict[str, object], model: L.LightningModule) -> None:
    optimizer_cfg = config.get("optimizer") or {}
    lr = optimizer_cfg.get("lr")
    weight_decay = optimizer_cfg.get("weight_decay")

    if lr is not None:
        model.lr = float(lr)  # type: ignore[attr-defined]
    if weight_decay is not None:
        model.weight_decay = float(weight_decay)  # type: ignore[attr-defined]


def train(config: Dict[str, object], args: argparse.Namespace) -> None:
    seed = config.get("seed")
    if seed is not None:
        L.seed_everything(int(seed), workers=True)

    datamodule = instantiate_datamodule(config)
    model = instantiate_model(config, datamodule)
    setup_optimizer_params(config, model)

    logging_config = config.get("logging") or {}
    checkpoint_dir = Path(config.get("checkpoint_dir", "artifacts/checkpoints"))

    callbacks = build_callbacks(config, checkpoint_dir)
    loggers = build_loggers(logging_config)

    trainer_cfg = config.get("trainer") or {}
    precision_kwargs = get_precision_kwargs(config.get("precision"))
    trainer = L.Trainer(
        max_epochs=int(trainer_cfg.get("max_epochs", 10)),
        accelerator=trainer_cfg.get("accelerator", "auto"),
        devices=trainer_cfg.get("devices", "auto"),
        gradient_clip_val=float(trainer_cfg.get("gradient_clip_val", 1.0)),
        gradient_clip_algorithm=trainer_cfg.get("gradient_clip_algorithm", "norm"),
        accumulate_grad_batches=int(trainer_cfg.get("accumulate_grad_batches", 1)),
        deterministic=trainer_cfg.get("deterministic", False),
        logger=loggers,
        callbacks=callbacks,
        default_root_dir=str(checkpoint_dir.parent),
        log_every_n_steps=int(trainer_cfg.get("log_every_n_steps", 50)),
        enable_checkpointing=True,
        fast_dev_run=args.fast_dev_run,
        **precision_kwargs,
    )

    ckpt_path = args.resume_from_checkpoint or trainer_cfg.get("resume_from_checkpoint")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CTC-based speech recognition model.")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Path to a YAML configuration file.")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="Resume training from a checkpoint.")
    parser.add_argument("--fast-dev-run", action="store_true", help="Run a single batch for debugging purposes.")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return safe_load(handle)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    train(config, args)


if __name__ == "__main__":
    main()
