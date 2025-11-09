#!/usr/bin/env python3
"""Benchmark OCR backbones on a validation set.

This script compares model predictions against a ground-truth annotation file
and reports Character Error Rate (CER), Word Error Rate (WER), and average
inference time per sample. It supports two modes:

1. Offline mode (default for this repository): predictions and timing metadata
   are read from disk. This is useful in environments without access to the
   heavy OCR model dependencies.
2. Online mode (not implemented here): users can extend the script to call
   actual OCR backbones once the required dependencies are available.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

AnnotationDict = Dict[str, str]
PredictionDict = Dict[str, str]


@dataclass
class BenchmarkResult:
    model_name: str
    cer: float
    wer: float
    average_latency: Optional[float]
    num_samples: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark OCR backbones")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/val/annotations.tsv"),
        help="Path to the TSV file containing filename and transcript columns.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("data/val/predictions"),
        help="Directory containing per-model prediction TSV files.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/val/predictions/metadata.json"),
        help="JSON file with timing metadata (optional).",
    )
    return parser.parse_args()


def load_annotations(path: Path) -> AnnotationDict:
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    annotations: AnnotationDict = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        expected = {"filename", "transcript"}
        if set(reader.fieldnames or []) != expected:
            raise ValueError(
                f"Annotation file must contain columns {expected}; got {reader.fieldnames}"
            )
        for row in reader:
            filename = row["filename"].strip()
            transcript = row["transcript"].strip()
            if not filename:
                continue
            annotations[filename] = transcript
    if not annotations:
        raise ValueError("No annotations loaded; check the dataset.")
    return annotations


def load_prediction_file(predictions_path: Path, annotations: AnnotationDict) -> PredictionDict:
    if not predictions_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {predictions_path}")

    predictions: PredictionDict = {}
    with predictions_path.open("r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) != 2:
                raise ValueError(
                    f"Prediction rows must have 2 columns (filename, prediction); got {row}"
                )
            filename, prediction = row
            filename = filename.strip()
            if filename in annotations:
                predictions[filename] = prediction.strip()
    missing = set(annotations) - set(predictions)
    if missing:
        raise ValueError(
            f"Missing predictions for {len(missing)} samples in {predictions_path.name}: {sorted(missing)}"
        )
    return predictions


def load_latency_from_metadata(metadata_path: Path, model_name: str, num_samples: int) -> Optional[float]:
    if not metadata_path.exists():
        return None
    with metadata_path.open("r") as f:
        data = json.load(f)
    if model_name not in data:
        return None
    entry = data[model_name]
    total_time = float(entry.get("total_inference_seconds", math.nan))
    reported_samples = int(entry.get("num_samples", num_samples)) or num_samples
    if not math.isfinite(total_time) or reported_samples <= 0:
        return None
    return total_time / reported_samples


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev_row = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current_row = [i]
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            current_row.append(
                min(
                    current_row[j - 1] + 1,  # insertion
                    prev_row[j] + 1,          # deletion
                    prev_row[j - 1] + cost,   # substitution
                )
            )
        prev_row = current_row
    return prev_row[-1]


def cer(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0
    return levenshtein_distance(reference, hypothesis) / len(reference)


def wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0
    return levenshtein_distance(" ".join(ref_words), " ".join(hyp_words)) / len(ref_words)


def compute_metrics(annotations: AnnotationDict, predictions: PredictionDict) -> Tuple[float, float]:
    cer_total = 0.0
    wer_total = 0.0
    for filename, ground_truth in annotations.items():
        predicted = predictions[filename]
        cer_total += cer(ground_truth, predicted)
        wer_total += wer(ground_truth, predicted)
    num_samples = len(annotations)
    return cer_total / num_samples, wer_total / num_samples


def evaluate_model(
    model_name: str,
    annotations: AnnotationDict,
    predictions_dir: Path,
    metadata_path: Path,
) -> BenchmarkResult:
    predictions = load_prediction_file(predictions_dir / f"{model_name}.txt", annotations)
    cer_value, wer_value = compute_metrics(annotations, predictions)
    avg_latency = load_latency_from_metadata(metadata_path, model_name, len(annotations))
    return BenchmarkResult(
        model_name=model_name,
        cer=cer_value,
        wer=wer_value,
        average_latency=avg_latency,
        num_samples=len(annotations),
    )


def format_latency(latency: Optional[float]) -> str:
    return f"{latency*1000:.1f} ms" if latency is not None else "N/A"


def main() -> None:
    args = parse_args()
    annotations = load_annotations(args.annotations)

    models = ["trocr", "donut", "paddleocr"]
    results: List[BenchmarkResult] = []
    for model in models:
        result = evaluate_model(model, annotations, args.predictions_dir, args.metadata)
        results.append(result)

    print("Model\tCER\tWER\tAvg latency")
    for result in results:
        print(
            f"{result.model_name}\t{result.cer:.4f}\t{result.wer:.4f}\t{format_latency(result.average_latency)}"
        )


if __name__ == "__main__":
    main()
