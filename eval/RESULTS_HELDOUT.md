# ColdLens Held-out Evaluation

10 images from `heldout/` (independent generator: foil blisters, inkjet dot-matrix stamps, curved vials/bottles, carton flaps, shadows, glare). Current default pipeline only: /analyze preprocessing, current ocr.py and extraction prompt.
Groq API usage this run: 10 fresh calls. Compliance flags evaluated as of 2026-10-10.

## Per-field accuracy

| Field | Strict | Fuzzy* |
|---|---|---|
| product | 40% | 50% |
| batch_no | 30% | 30% |
| mfg_date | 30% | 30% |
| expiry_date | 70% | 70% |
| storage_temp_min | 100% | 100% |
| storage_temp_max | 70% | 70% |
| **Overall** | **57%** | **58%** |

*fuzzy = strict for every field except product (rapidfuzz ratio >= 85).

## Per-style breakdown

| Style | n | Overall (strict) | Overall (fuzzy) | Flags correct |
|---|---|---|---|---|
| blister | 2 | 83% | 83% | 2/2 |
| bottle | 2 | 42% | 50% | 1/2 |
| carton_flap | 2 | 25% | 25% | 0/2 |
| side_panel | 2 | 50% | 50% | 1/2 |
| vial | 2 | 83% | 83% | 2/2 |

## Every mismatch (26 of 60 fields, strict)

| Image | Style | Field | Expected | Extracted | Fuzzy OK? |
|---|---|---|---|---|---|
| heldout_01 | blister | mfg_date | 2025-03-12 | 2025-12-03 | - |
| heldout_02 | carton_flap | product | Azithromycin 500 mg Tablets | Azithromycin | no (61.5) |
| heldout_02 | carton_flap | batch_no | AZM5L042 | AZMSL84Z | - |
| heldout_02 | carton_flap | mfg_date | 2025-06-04 | None | - |
| heldout_02 | carton_flap | expiry_date | 2027-05-31 | 2025-06-04 | - |
| heldout_04 | bottle | product | Cetirizine Oral Solution | Cetirizine | no (58.8) |
| heldout_04 | bottle | batch_no | CTZ-1173 | None | - |
| heldout_04 | bottle | mfg_date | 2025-01-20 | None | - |
| heldout_04 | bottle | expiry_date | 2027-12-31 | None | - |
| heldout_04 | bottle | storage_temp_max | 30 | None | - |
| heldout_05 | side_panel | product | Metformin 500 mg Tablets | Metformin 500 mg | no (80.0) |
| heldout_05 | side_panel | batch_no | MTF0925B | MTFO925B | - |
| heldout_05 | side_panel | mfg_date | 2025-09-02 | 2025-02-09 | - |
| heldout_06 | blister | mfg_date | 2024-11-08 | 2024-08-11 | - |
| heldout_07 | vial | batch_no | IGL7F203 | IGLZF203 | - |
| heldout_07 | vial | mfg_date | 2025-07-03 | 2025-03-07 | - |
| heldout_08 | carton_flap | product | Amoxicillin 250 mg Capsules | None | no (0.0) |
| heldout_08 | carton_flap | batch_no | AMX3C551 | None | - |
| heldout_08 | carton_flap | mfg_date | 2026-02-14 | None | - |
| heldout_08 | carton_flap | expiry_date | 2028-01-31 | None | - |
| heldout_08 | carton_flap | storage_temp_max | 25 | None | - |
| heldout_09 | bottle | product | Oral Rehydration Salts Solution | Rehydration Salts Solution | yes (91.2) |
| heldout_09 | bottle | batch_no | ORS2208K | ORS22O8K | - |
| heldout_10 | side_panel | product | Pantoprazole 40 mg Tablets | Pantoprazole 40mg | no (79.1) |
| heldout_10 | side_panel | batch_no | PNT40E19 | PNT A0E19 | - |
| heldout_10 | side_panel | storage_temp_max | 30 | 0.0 | - |

## Compliance flags (6/10 images fully correct)

Expected = the same compliance rules applied to the ground-truth fields.

| Image | Style | Expected flags | Got flags | Correct? |
|---|---|---|---|---|
| heldout_01 | blister | none | none | yes |
| heldout_02 | carton_flap | none | expired | NO - expired |
| heldout_03 | vial | cold_storage_required | cold_storage_required | yes |
| heldout_04 | bottle | none | missing_batch_no | NO - missing_batch_no |
| heldout_05 | side_panel | none | none | yes |
| heldout_06 | blister | expiring_soon | expiring_soon | yes |
| heldout_07 | vial | cold_storage_required | cold_storage_required | yes |
| heldout_08 | carton_flap | none | missing_batch_no | NO - missing_batch_no |
| heldout_09 | bottle | none | none | yes |
| heldout_10 | side_panel | expired | expired, cold_storage_required | NO - cold_storage_required |
