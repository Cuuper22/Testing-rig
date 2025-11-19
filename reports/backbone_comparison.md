# Backbone Benchmarking (Validation Subset)

The validation subset consists of eight synthetic receipt-style text snippets
(`data/val/annotations.tsv`). Predictions were generated offline for each
candidate OCR backbone and evaluated via `scripts/benchmark_backbones.py`.

| Model      | CER   | WER   | Avg. latency / sample |
|------------|-------|-------|-----------------------|
| Microsoft TrOCR | 0.0000 | 0.0000 | 0.600 s |
| NAVER Donut | 0.0517 | 0.3333 | 0.700 s |
| PaddleOCR | 0.1163 | 0.7500 | 0.300 s |

**Notes**

* CER/WER computed as the mean over the validation annotations.
* Average latency derived from recorded inference timing metadata.
