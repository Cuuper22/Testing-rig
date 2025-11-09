# Parallel Coptic-English Corpora

This directory is expected to contain sentence-level parallel corpora derived from the Coptic Scriptorium project. Each split should be stored as a UTF-8 encoded TSV file with the following names and format:

- `train.tsv`
- `dev.tsv`
- `test.tsv`

Each file must have a header row: `coptic\tenglish`. Text should be normalized to NFC and stripped of leading or trailing whitespace. See `scripts/download_coptic_corpus.py` for an automated download workflow.
