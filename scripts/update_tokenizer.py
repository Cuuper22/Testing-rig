#!/usr/bin/env python3
"""Train and update the tokenizer to cover the full Coptic Unicode block.

This script trains a tokenizer on the provided transcription corpus, ensures
that all code points in the Coptic Unicode block (U+2C80–U+2CFF) are present in
its vocabulary, saves the tokenizer artefacts, and evaluates coverage on a
validation split.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

try:
    from tokenizers import Tokenizer, normalizers, pre_tokenizers
    from tokenizers.models import BPE, Unigram
    from tokenizers.processors import TemplateProcessing
    from tokenizers.trainers import BpeTrainer, UnigramTrainer
    HAS_TOKENIZERS = True
except ModuleNotFoundError:  # pragma: no cover - executed only in constrained envs
    HAS_TOKENIZERS = False

    @dataclass
    class Encoding:
        tokens: List[str]

    class SimpleTokenizer:
        """Character-level fallback tokenizer used when `tokenizers` is unavailable."""

        def __init__(self, vocab: Sequence[str]):
            self._tokens: List[str] = list(dict.fromkeys(vocab))
            self._id_for_token = {token: idx for idx, token in enumerate(self._tokens)}

        def add_tokens(self, tokens: Sequence[str]) -> int:
            added = 0
            for token in tokens:
                if token not in self._id_for_token:
                    self._id_for_token[token] = len(self._tokens)
                    self._tokens.append(token)
                    added += 1
            return added

        def get_vocab(self) -> dict:
            return dict(self._id_for_token)

        def encode(self, text: str) -> Encoding:
            tokens = []
            for char in text:
                tokens.append(char if char in self._id_for_token else "[UNK]")
            return Encoding(tokens)

        def token_to_id(self, token: str) -> int | None:
            return self._id_for_token.get(token)

        def save(self, path: str) -> None:
            data = {
                "type": "character",
                "tokens": self._tokens,
            }
            Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

        # Attributes kept for API parity with `tokenizers.Tokenizer`
        normalizer = None
        pre_tokenizer = None
        post_processor = None

    # Create shims so the rest of the module can stay largely unchanged.
    Tokenizer = SimpleTokenizer  # type: ignore
    BPE = Unigram = object  # placeholders, not used in fallback mode
    TemplateProcessing = None  # type: ignore

    class _FakeTrainer:
        def __init__(self, *_, **__):
            self.show_progress = False

    BpeTrainer = UnigramTrainer = _FakeTrainer  # type: ignore


COPTIC_BLOCK = range(0x2C80, 0x2D00)  # exclusive end
LEGACY_COPTIC_BLOCK = range(0x03E2, 0x03F0)  # covers U+03E2–U+03EF
SPECIAL_TOKENS = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']


def iter_lines(paths: List[Path]) -> Iterable[str]:
    """Yield non-empty, stripped lines from a collection of text files."""
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    yield text


def train_tokenizer(train_files: List[Path], model_type: str, vocab_size: int):
    """Train either a BPE or Unigram tokenizer on the training corpus."""
    if HAS_TOKENIZERS:
        if model_type == "bpe":
            model = BPE(unk_token="[UNK]")
            trainer = BpeTrainer(
                vocab_size=vocab_size,
                min_frequency=2,
                special_tokens=SPECIAL_TOKENS,
            )
        elif model_type == "unigram":
            model = Unigram(unk_token="[UNK]")
            trainer = UnigramTrainer(
                vocab_size=vocab_size,
                special_tokens=SPECIAL_TOKENS,
                unk_token="[UNK]",
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        tokenizer = Tokenizer(model)
        tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFD(),
            normalizers.Lowercase(),
            normalizers.StripAccents(),
            normalizers.NFC(),
        ])
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Whitespace(),
            pre_tokenizers.Punctuation(),
        ])
        trainer.show_progress = False

        tokenizer.train_from_iterator(iter_lines(train_files), trainer=trainer)

        tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B [SEP]",
            special_tokens=[
                ("[CLS]", tokenizer.token_to_id("[CLS]")),
                ("[SEP]", tokenizer.token_to_id("[SEP]")),
            ],
        )
        return tokenizer

    # Fallback: build a simple character-level tokenizer.
    corpus_chars = set().union(*[set(line) for line in iter_lines(train_files)])
    vocab = list(SPECIAL_TOKENS) + sorted(corpus_chars - set(SPECIAL_TOKENS))
    return Tokenizer(vocab)


def add_coptic_codepoints(tokenizer) -> int:
    """Ensure all Coptic characters are present in the tokenizer vocabulary."""
    current_vocab = tokenizer.get_vocab()
    missing_tokens = [
        chr(code_point)
        for code_point in list(COPTIC_BLOCK) + list(LEGACY_COPTIC_BLOCK)
        if chr(code_point) not in current_vocab
    ]

    if not missing_tokens:
        return 0

    tokenizer.add_tokens(missing_tokens)
    return len(missing_tokens)


def evaluate_coverage(tokenizer, validation_files: List[Path]) -> dict:
    """Compute coverage statistics for the validation corpus."""
    total_tokens = 0
    unknown_tokens = 0
    line_count = 0

    for line in iter_lines(validation_files):
        encoding = tokenizer.encode(line)
        tokens = encoding.tokens if hasattr(encoding, "tokens") else encoding.tokens
        total_tokens += len(tokens)
        unknown_tokens += sum(token == "[UNK]" for token in tokens)
        line_count += 1

    coverage = {
        "lines": line_count,
        "total_tokens": total_tokens,
        "unknown_tokens": unknown_tokens,
        "unknown_ratio": 0.0 if total_tokens == 0 else unknown_tokens / total_tokens,
    }
    return coverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        nargs="+",
        required=True,
        type=Path,
        help="Training transcription files.",
    )
    parser.add_argument(
        "--validation",
        nargs="+",
        required=True,
        type=Path,
        help="Validation transcription files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the tokenizer artefacts will be stored.",
    )
    parser.add_argument(
        "--coverage-report",
        type=Path,
        required=True,
        help="Path to the JSON coverage report.",
    )
    parser.add_argument(
        "--model",
        choices=["bpe", "unigram"],
        default="bpe",
        help="Type of tokenizer model to train.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=1024,
        help="Target vocabulary size for the tokenizer trainer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = train_tokenizer(args.train, args.model, args.vocab_size)
    added = add_coptic_codepoints(tokenizer)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = args.output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    coverage = evaluate_coverage(tokenizer, args.validation)
    coverage["added_coptic_tokens"] = added
    coverage["tokenizer_path"] = str(tokenizer_path)
    coverage["uses_tokenizers_library"] = HAS_TOKENIZERS

    if coverage["unknown_tokens"] != 0:
        raise RuntimeError(
            "Validation corpus still contains unknown tokens after the update."
        )

    args.coverage_report.parent.mkdir(parents=True, exist_ok=True)
    with args.coverage_report.open("w", encoding="utf-8") as handle:
        json.dump(coverage, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
