# ColdLens Evaluation Results

Synthetic dataset: 2 images x 4 preprocessing variants.
Groq API usage this run: 0 fresh calls, 0 cache hits.

## Overall accuracy per variant

Overall accuracy = mean of the 6 per-field accuracies (macro average), not "all 6 fields exactly right."

| Variant | Overall | product | batch_no | mfg_date | expiry_date | storage_temp_min | storage_temp_max |
|---|---|---|---|---|---|---|---|
| no_preprocessing | **67%** | 0% | 100% | 100% | 100% | 50% | 50% |
| grayscale_denoise | **67%** | 0% | 100% | 100% | 100% | 50% | 50% |
| full_no_threshold | **67%** | 0% | 100% | 100% | 100% | 50% | 50% |
| full_pipeline | **83%** | 0% | 100% | 100% | 100% | 100% | 100% |

## Accuracy by distortion type (overall, per variant)

| Distortion | no_preprocessing | grayscale_denoise | full_no_threshold | full_pipeline |
|---|---|---|---|---|
| blur | 50% (n=1) | 50% (n=1) | 50% (n=1) | 83% (n=1) |
| none | 83% (n=1) | 83% (n=1) | 83% (n=1) | 83% (n=1) |
