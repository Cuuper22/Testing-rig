from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from .preprocessing import ScalerParams


@dataclass
class RunInfo:
    """Information describing a single training run."""

    run_id: str
    run_name: str
    path: Path


class ModelRegistry:
    """Persist training artefacts, metrics and tokenizer configuration."""

    def __init__(self, base_path: str | Path = "mlruns") -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._index_file = self.base_path / "registry.json"
        if not self._index_file.exists():
            self._index_file.write_text(json.dumps({}, indent=2))

    def _load_index(self) -> dict[str, list[str]]:
        raw = json.loads(self._index_file.read_text())
        return {name: list(runs.keys()) for name, runs in raw.items()}

    def _write_index(self, index: dict[str, list[str]]) -> None:
        payload = {
            name: {run_id: "meta.json" for run_id in runs}
            for name, runs in index.items()
        }
        self._index_file.write_text(json.dumps(payload, indent=2))

    def start_run(self, run_name: str) -> str:
        index = self._load_index()
        run_id = uuid4().hex
        run_dir = self.base_path / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        (run_dir / "meta.json").write_text(json.dumps({"run_name": run_name}, indent=2))

        runs = index.setdefault(run_name, [])
        runs.append(run_id)
        self._write_index(index)
        return run_id

    def log_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        run_dir = self.base_path / run_id
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    def log_model(
        self, run_id: str, weights: list[float], bias: float, scaler: ScalerParams
    ) -> None:
        run_dir = self.base_path / run_id
        model_payload = {
            "weights": weights,
            "bias": bias,
            "scaler": {
                "mean": scaler.mean,
                "std": scaler.std,
            },
        }
        (run_dir / "model.json").write_text(json.dumps(model_payload, indent=2))

    def log_tokenizer(self, run_id: str, tokenizer: dict[str, int]) -> None:
        run_dir = self.base_path / run_id
        (run_dir / "tokenizer.json").write_text(json.dumps(tokenizer, indent=2))

    def get_latest_run(self, run_name: str) -> RunInfo | None:
        index = self._load_index()
        runs = index.get(run_name)
        if not runs:
            return None
        latest_run_id = max(
            runs, key=lambda run_id: (self.base_path / run_id).stat().st_mtime
        )
        run_dir = self.base_path / latest_run_id
        return RunInfo(run_id=latest_run_id, run_name=run_name, path=run_dir)

    def load_model(self, run_id: str) -> tuple[list[float], float, ScalerParams]:
        run_dir = self.base_path / run_id
        payload = json.loads((run_dir / "model.json").read_text())
        scaler_payload = payload["scaler"]
        scaler = ScalerParams(
            mean=list(scaler_payload["mean"]),
            std=list(scaler_payload["std"]),
        )
        return list(payload["weights"]), float(payload["bias"]), scaler

    def load_metrics(self, run_id: str) -> dict[str, float]:
        run_dir = self.base_path / run_id
        return json.loads((run_dir / "metrics.json").read_text())

    def load_tokenizer(self, run_id: str) -> dict[str, int] | None:
        run_dir = self.base_path / run_id
        tokenizer_file = run_dir / "tokenizer.json"
        if not tokenizer_file.exists():
            return None
        return json.loads(tokenizer_file.read_text())
