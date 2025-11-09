"""Fine-tune a sequence-to-sequence model for Coptic→English translation."""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)


@dataclass
class ParallelExample:
    coptic: str
    english: str


SUPPORTED_MODELS = {
    "marian": "Helsinki-NLP/opus-mt-cop-en",
    "mt5": "google/mt5-small",
}


def read_parallel_tsv(path: Path) -> List[ParallelExample]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        missing_cols = {"coptic", "english"} - set(reader.fieldnames or [])
        if missing_cols:
            raise ValueError(f"Missing columns {missing_cols} in {path}")
        rows = [ParallelExample(row["coptic"].strip(), row["english"].strip()) for row in reader]
    return rows


def make_dataset(path: Path) -> Dataset:
    examples = read_parallel_tsv(path)
    return Dataset.from_dict({
        "coptic": [ex.coptic for ex in examples],
        "english": [ex.english for ex in examples],
    })


def load_datasets(train_path: Path, dev_path: Path | None, test_path: Path | None) -> Dict[str, Dataset]:
    datasets: Dict[str, Dataset] = {"train": make_dataset(train_path)}
    if dev_path and dev_path.exists():
        datasets["validation"] = make_dataset(dev_path)
    if test_path and test_path.exists():
        datasets["test"] = make_dataset(test_path)
    return datasets


def preprocess_function(tokenizer, max_source_length: int, max_target_length: int):
    def inner(examples):
        model_inputs = tokenizer(
            examples["coptic"],
            max_length=max_source_length,
            padding="max_length",
            truncation=True,
        )
        labels = tokenizer(
            examples["english"],
            max_length=max_target_length,
            padding="max_length",
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return inner


def compute_bleu_metrics(tokenizer, predictions, labels):
    try:
        import sacrebleu
    except ImportError as exc:  # pragma: no cover - metric import guard
        raise RuntimeError("sacrebleu must be installed for evaluation") from exc

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
    chrf = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels])
    return {"bleu": bleu.score, "chrf": chrf.score}



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/parallel"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=sorted(SUPPORTED_MODELS.keys()),
        default="mt5",
        help="Base model family to fine-tune.",
    )
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--num-train-epochs", type=float, default=5.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id")
    parser.add_argument("--hub-token")
    parser.add_argument("--report-to", nargs="*", default=["tensorboard"])
    parser.add_argument("--num-beams", type=int, default=4)
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    set_seed(args.seed)

    train_path = args.data_dir / "train.tsv"
    dev_path = args.data_dir / "dev.tsv"
    test_path = args.data_dir / "test.tsv"

    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")

    datasets = load_datasets(train_path, dev_path, test_path)

    model_name = SUPPORTED_MODELS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    preprocess = preprocess_function(
        tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )

    tokenized_datasets = {
        split: dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
        for split, dataset in datasets.items()
    }

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        evaluation_strategy="steps" if "validation" in tokenized_datasets else "no",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=100,
        warmup_steps=args.warmup_steps,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        generation_num_beams=args.num_beams,
        fp16=args.fp16,
        report_to=args.report_to,
        load_best_model_at_end="validation" in tokenized_datasets,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_token=args.hub_token,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        return compute_bleu_metrics(tokenizer, preds, labels)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("validation"),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if "validation" in tokenized_datasets else None,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if "test" in tokenized_datasets:
        logger.info("Running test evaluation...")
        test_metrics = trainer.evaluate(tokenized_datasets["test"], metric_key_prefix="test")
        for key, value in test_metrics.items():
            logger.info("test_%s = %.4f", key, value)


if __name__ == "__main__":
    main()
