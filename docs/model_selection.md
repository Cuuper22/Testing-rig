# Model Selection Rationale

The benchmarking results (see `reports/backbone_comparison.md`) show a clear
accuracy gap between the candidates:

* **Microsoft TrOCR** achieved zero character and word errors on the validation
  subset with an average latency of 0.60 seconds per sample.
* **NAVER Donut** reduced speed slightly (0.70 seconds per sample) but
  introduced punctuation and delimiter mistakes that raised CER to 5.17% and WER
  to 33.33%.
* **PaddleOCR** was the fastest at 0.30 seconds per sample but suffered the
  highest error rates (11.63% CER / 75.00% WER), largely from dropping
  punctuation.

Given the product requirement for accurate receipt transcription, Microsoft
TrOCR provides the best trade-off: it eliminates post-correction costs while
remaining within acceptable latency bounds for offline batch processing. The
extra 0.30 seconds per sample versus PaddleOCR is outweighed by the substantial
accuracy gains and reduced downstream validation effort.
