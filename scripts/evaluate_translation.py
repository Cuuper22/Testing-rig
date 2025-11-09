#!/usr/bin/env python
"""Evaluate a trained translation model using BLEU/ChrF and qualitative examples."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import sacrebleu

from src.translation.pipeline import PipelineConfig, TranslationPipeline


def load_parallel(path: Path) -> List[Tuple[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None or set(reader.fieldnames) != {"coptic", "english"}:
            raise ValueError(f"Expected TSV header 'coptic\tenglish' in {path}")
        return [(row["coptic"].strip(), row["english"].strip()) for row in reader]


def compute_metrics(predictions: List[str], references: List[str]) -> Tuple[float, float]:
    bleu = sacrebleu.corpus_bleu(predictions, [references]).score
    chrf = sacrebleu.corpus_chrf(predictions, [references]).score
    return bleu, chrf


def write_report(
    output_path: Path,
    bleu: float,
    chrf: float,
    examples: List[Tuple[str, str, str]],
) -> None:
    lines = ["# Machine Translation Evaluation", ""]
    lines.append(f"* **BLEU**: {bleu:.2f}")
    lines.append(f"* **ChrF**: {chrf:.2f}")
    lines.append("\n## Qualitative Examples\n")
    for idx, (source, prediction, reference) in enumerate(examples, start=1):
        lines.append(f"### Example {idx}")
        lines.append("")
        lines.append(f"- **Coptic (normalized)**: `{source}`")
        lines.append(f"- **Prediction**: `{prediction}`")
        lines.append(f"- **Reference**: `{reference}`")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/parallel/test.tsv"))
    parser.add_argument("--report", type=Path, default=Path("reports/translation_eval.md"))
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = load_parallel(args.dataset)
    if not pairs:
        raise RuntimeError(f"No examples found in {args.dataset}")

    config = PipelineConfig(
        model_path=str(args.model_path),
        num_beams=args.num_beams,
        max_length=args.max_length,
    )
    pipeline = TranslationPipeline(config)

    sources = [src for src, _ in pairs]
    references = [tgt for _, tgt in pairs]
    predictions = pipeline.translate(sources)

    bleu, chrf = compute_metrics(predictions, references)

    qualitative_examples = []
    for source, prediction, reference in zip(sources, predictions, references):
        qualitative_examples.append((pipeline.normalize_text(source), prediction, reference))
        if len(qualitative_examples) >= args.max_examples:
            break

    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.report, bleu, chrf, qualitative_examples)


if __name__ == "__main__":
    main()
