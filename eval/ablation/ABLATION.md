# OCR ablation (full_no_threshold, new prompt)

Fresh Groq calls this run: 75.

baseline = old ocr.py + old prompt (results_baseline.csv).

| Field | baseline | a_old_ocr | b_rows_no_conf | c_current |
|---|---|---|---|---|
| product | 46% | 64% | 72% | 68% |
| batch_no | 74% | 78% | 74% | 74% |
| mfg_date | 70% | 66% | 66% | 56% |
| expiry_date | 82% | 70% | 68% | 60% |
| storage_temp_min | 86% | 84% | 80% | 74% |
| storage_temp_max | 82% | 76% | 74% | 64% |
| **Overall (strict)** | **73%** | **73%** | **72%** | **66%** |
| **Overall (fuzzy product)** | **79%** | **75%** | **73%** | **67%** |
