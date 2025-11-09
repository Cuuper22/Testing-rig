# Testing-rig

This repository demonstrates how to update and retrain a tokenizer with full
coverage of the Coptic Unicode repertoire.

## Dataset

Sample training and validation transcriptions are stored under
`data/transcriptions/`. They include characters from both the modern Coptic
block (U+2C80–U+2CFF) and the legacy Greek/Coptic block (U+03E2–U+03EF).

## Usage

Run the update script to retrain the tokenizer, ensure the vocabulary contains
all Coptic code points, and validate coverage on the held-out split:

```bash
python scripts/update_tokenizer.py \
  --train data/transcriptions/train.txt \
  --validation data/transcriptions/val.txt \
  --output-dir artifacts/tokenizer \
  --coverage-report reports/tokenizer_coverage.json \
  --model unigram \
  --vocab-size 512
```

The script prefers the [`tokenizers`](https://github.com/huggingface/tokenizers)
library. If the dependency is unavailable, it falls back to a
character-level trainer that still guarantees Coptic coverage.

Outputs:

- `artifacts/tokenizer/tokenizer.json`: serialized tokenizer.
- `reports/tokenizer_coverage.json`: coverage statistics (should report zero
  unknown tokens on the validation set).
