# ColdLens Evaluation Results

Synthetic dataset: 50 images x 4 preprocessing variants.
Groq API usage this run: 111 fresh calls, 89 cache hits.


## Before vs. after improvements (full_no_threshold)

Before = pre-fix ocr.py (raw EasyOCR detection order, no confidence filter) and extract.py (no OCR character-confusion correction in the prompt). After = this run - ocr.py now drops lines under 0.3 confidence and reorders text into rows (top-to-bottom, left-to-right) before handing it to the LLM, and the prompt now explicitly corrects O/0, S/5, l-I/1 confusions in numeric contexts.

| Field | Before (strict) | After (strict) | Before (fuzzy*) | After (fuzzy*) |
|---|---|---|---|---|
| product | 46% | 66% | 80% | 70% |
| batch_no | 74% | 72% | 74% | 72% |
| mfg_date | 70% | 54% | 70% | 54% |
| expiry_date | 82% | 60% | 82% | 60% |
| storage_temp_min | 86% | 74% | 86% | 74% |
| storage_temp_max | 82% | 64% | 82% | 64% |
| **Overall** | **73%** | **65%** | **79%** | **66%** |

*fuzzy = strict match for every field except product, which uses rapidfuzz ratio >= 85.


## Overall accuracy per variant (strict exact match)

Overall accuracy = mean of the 6 per-field accuracies (macro average), not "all 6 fields exactly right." Product name is particularly harsh under strict matching - see the fuzzy-matching section below.

| Variant | Overall | product | batch_no | mfg_date | expiry_date | storage_temp_min | storage_temp_max |
|---|---|---|---|---|---|---|---|
| no_preprocessing | **58%** | 56% | 56% | 54% | 58% | 70% | 56% |
| grayscale_denoise | **66%** | 62% | 68% | 56% | 58% | 78% | 74% |
| full_no_threshold | **65%** | 66% | 72% | 54% | 60% | 74% | 64% |
| full_pipeline | **64%** | 62% | 58% | 50% | 62% | 84% | 70% |

## Product name: strict vs. fuzzy matching

EasyOCR routinely confuses digits/letters in the bold title font (e.g. "500mg" -> "50Omg"/"SOOmg"), even on undistorted images - a 1-character miss out of ~25 that strict exact-match scores as a total failure. Fuzzy match = `rapidfuzz.fuzz.ratio >= 85` on the case/whitespace-normalized strings.

| Variant | Product (strict) | Product (fuzzy) | Overall (strict) | Overall (fuzzy product) |
|---|---|---|---|---|
| no_preprocessing | 56% | 60% | 58% | 59% |
| grayscale_denoise | 62% | 66% | 66% | 67% |
| full_no_threshold | 66% | 70% | 65% | 66% |
| full_pipeline | 62% | 72% | 64% | 66% |

### Sample product near-misses (fails strict, passes fuzzy)

| Image | Variant | Ground truth | Extracted | Fuzzy ratio |
|---|---|---|---|---|
| label_003 | no_preprocessing | Ciprofloxacin 500mg Tablets | Ciprofloxacin 50mg Tablets | 98.1 |
| label_003 | grayscale_denoise | Ciprofloxacin 500mg Tablets | Ciprofloxacin 50mg Tablets | 98.1 |
| label_003 | full_no_threshold | Ciprofloxacin 500mg Tablets | Ciprofloxacin 50mg Tablets | 98.1 |
| label_003 | full_pipeline | Ciprofloxacin 500mg Tablets | Ciprofloxacin 0.5mg Tablets | 92.6 |
| label_012 | full_pipeline | Ciprofloxacin 500mg Tablets | Ciprofloxacin 100mg Tablets | 96.3 |

### Sample product mismatches (fails even fuzzy)

| Image | Variant | Ground truth | Extracted | Fuzzy ratio |
|---|---|---|---|---|
| label_002 | no_preprocessing | Metformin 500mg Tablets | None | 0.0 |
| label_002 | full_pipeline | Metformin 500mg Tablets | None | 0.0 |
| label_004 | no_preprocessing | Doxycycline 100mg Capsules | None | 0.0 |
| label_004 | grayscale_denoise | Doxycycline 100mg Capsules | None | 0.0 |
| label_004 | full_no_threshold | Doxycycline 100mg Capsules | None | 0.0 |

## Accuracy by distortion type (overall, per variant)

| Distortion | no_preprocessing | grayscale_denoise | full_no_threshold | full_pipeline |
|---|---|---|---|---|
| blur | 43% (n=9) | 48% (n=9) | 56% (n=9) | 54% (n=9) |
| glare | 56% (n=8) | 67% (n=8) | 67% (n=8) | 71% (n=8) |
| low_light | 42% (n=6) | 58% (n=6) | 58% (n=6) | 42% (n=6) |
| noise | 83% (n=6) | 83% (n=6) | 83% (n=6) | 69% (n=6) |
| none | 78% (n=3) | 72% (n=3) | 72% (n=3) | 72% (n=3) |
| perspective | 81% (n=9) | 78% (n=9) | 76% (n=9) | 74% (n=9) |
| rotation | 41% (n=9) | 63% (n=9) | 52% (n=9) | 69% (n=9) |
