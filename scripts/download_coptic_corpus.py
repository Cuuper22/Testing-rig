#!/usr/bin/env python
"""Download and prepare Coptic Scriptorium parallel corpora."""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import Path
from typing import Iterable, Tuple

import requests

# Known releases of the Coptic Scriptorium project that contain aligned
# translations. The archive URLs reference the GitHub repository snapshots to
# avoid scraping the project website.
CORPUS_ARCHIVES = {
    "apostolic_fathers": "https://github.com/CopticScriptorium/corpora/archive/refs/heads/master.zip",
}

# Files within the archive that contain parallel lines. Each tuple maps the
# archive member to a pair of (coptic_file, english_file). These files are
# derived from the `parallel/` folder in the corpus repository.
PARALLEL_FILE_PAIRS = [
    (
        "corpora-master/parallel/Athanasius_apologia_secunda/apol_sec_coptic.txt",
        "corpora-master/parallel/Athanasius_apologia_secunda/apol_sec_english.txt",
    ),
]


def iter_parallel_lines(zf: zipfile.ZipFile, coptic_path: str, english_path: str) -> Iterable[Tuple[str, str]]:
    with zf.open(coptic_path) as coptic_fp, zf.open(english_path) as english_fp:
        coptic_lines = [line.decode("utf-8").strip() for line in coptic_fp]
        english_lines = [line.decode("utf-8").strip() for line in english_fp]
    if len(coptic_lines) != len(english_lines):
        raise ValueError(
            f"Mismatched line counts for {coptic_path} and {english_path}: "
            f"{len(coptic_lines)} vs {len(english_lines)}"
        )
    for cop, eng in zip(coptic_lines, english_lines):
        if not cop or not eng:
            continue
        yield cop, eng


def build_splits(pairs: Iterable[Tuple[str, str]], train_ratio: float, dev_ratio: float):
    pairs = list(pairs)
    total = len(pairs)
    train_end = int(total * train_ratio)
    dev_end = train_end + int(total * dev_ratio)
    return (
        pairs[:train_end],
        pairs[train_end:dev_end],
        pairs[dev_end:],
    )


def write_tsv(path: Path, rows: Iterable[Tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["coptic", "english"])
        for cop, eng in rows:
            writer.writerow([cop, eng])


def download_and_prepare(output_dir: Path, train_ratio: float = 0.8, dev_ratio: float = 0.1) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    collected_pairs = []
    for _, url in CORPUS_ARCHIVES.items():
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            for coptic_path, english_path in PARALLEL_FILE_PAIRS:
                collected_pairs.extend(iter_parallel_lines(zf, coptic_path, english_path))
    if not collected_pairs:
        raise RuntimeError("No parallel sentence pairs were extracted. Check archive definitions.")
    train_rows, dev_rows, test_rows = build_splits(collected_pairs, train_ratio, dev_ratio)
    write_tsv(output_dir / "train.tsv", train_rows)
    write_tsv(output_dir / "dev.tsv", dev_rows)
    write_tsv(output_dir / "test.tsv", test_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/parallel"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_and_prepare(args.output_dir, args.train_ratio, args.dev_ratio)


if __name__ == "__main__":
    main()
