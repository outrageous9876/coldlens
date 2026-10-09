# Prompt ablation (restored OCR, full_no_threshold)

Fresh Groq calls this run: 87 (65088 tokens).

| Field | baseline (committed run) | old_prompt | narrow_prompt |
|---|---|---|---|
| product | 46% | 42% | 80% |
| batch_no | 74% | 72% | 76% |
| mfg_date | 70% | 68% | 68% |
| expiry_date | 82% | 78% | 78% |
| storage_temp_min | 86% | 84% | 84% |
| storage_temp_max | 82% | 80% | 76% |
| product (fuzzy) | 80% | 76% | 82% |
| **Overall (strict)** | **73%** | **71%** | **77%** |
| **Overall (fuzzy product)** | **79%** | **76%** | **77%** |

Narrow rule tested:

```
- OCR often misreads digits in the dose inside the product name (e.g. "50Omg" or "SOOmg" is really "500mg"). In that dose ONLY, read O/o as 0 and S as 5. Do not apply this correction to any other field - leave batch numbers, dates and temperatures exactly as the rules below say.
```
