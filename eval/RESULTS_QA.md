# ColdLens Label Q&A Evaluation

10 questions over the Lantus, amoxicillin and metformin openFDA labels, asked through `POST /ask` (model `openai/gpt-oss-120b`, top-4 retrieved chunks). Questions, expected answers and source quotes: `eval/qa/questions.json`. Groq tokens this run: 0 (cached answers cost 0).

## Summary

| Metric | Score |
|---|---|
| Answer correct (in-label) | 7/7 |
| Right section cited (in-label) | 7/7 |
| Correctly said "not found" (out-of-label) | 3/3 |

Answer correct = said found AND every key fact from the label appears in the answer (acceptable phrasings per fact in questions.json).

## Per question

| ID | Product | Question | Correct? | Right section? | Cited | Missing key facts |
|---|---|---|---|---|---|---|
| q01 | Lantus | How should this be stored? | yes | yes | storage_and_handling | - |
| q02 | Lantus | Can it be frozen? | yes | yes | storage_and_handling | - |
| q03 | Lantus | Can Lantus be mixed with other insulins? | yes | yes | dosage_and_administration | - |
| q04 | amoxicillin | How long can the mixed oral suspension be kept? | yes | yes | dosage_and_administration | - |
| q05 | amoxicillin | What are the most common side effects? | yes | yes | adverse_reactions | - |
| q06 | metformin | Who should not take metformin? | yes | yes | contraindications | - |
| q07 | metformin | What serious risk does the label warn about? | yes | yes | warnings | - |
| q08 | Lantus | What is the price? | yes | - | - | - |
| q09 | amoxicillin | Who invented amoxicillin? | yes | - | - | - |
| q10 | metformin | Is metformin safe to give to my dog? | yes | - | - | - |

## Answers vs. expected

### q01: How should this be stored? (Lantus)

- **Expected:** Unopened: refrigerate at 36-46°F (2-8°C), do not freeze. Opened vial: refrigerated or room temperature (up to 86°F/30°C) for 28 days. Opened SoloStar pen: room temperature only, do not refrigerate, 28 days.
- **Label says:** "Store unused LANTUS in a refrigerator between 36°F and 46°F (2°C and 8°C). Do not freeze. ... 10 mL multiple-dose vial Until expiration date 28 days 28 days Refrigerated or room temperature 3 mL single-patient-use SoloStar prefilled pen Until expiration date 28 days 28 days Room temperature only (Do not refrigerate)"
- **Got:** 10 mL multiple‑dose vial: keep unopened in the refrigerator (36‑46°F/2‑8°C) until the expiration date; after first use it may be stored refrigerated for up to 28 days or at room temperature (up to 86°F/30°C) for up to 28 days. 3 mL SoloStar prefilled pen: keep unopened in the refrigerator (36‑46°F/2‑8°C) until the expiration date; after first use it may be stored for up to 28 days at room temperature only (do not refrigerate).
- **Matched label:** Lantus Solostar (sanofi-aventis U.S. LLC)

### q02: Can it be frozen? (Lantus)

- **Expected:** No. Do not freeze; discard Lantus if it has been frozen.
- **Label says:** "Do not freeze. Discard LANTUS if it has been frozen."
- **Got:** No, LANTUS must not be frozen; discard it if it has been frozen.
- **Matched label:** Lantus Solostar (sanofi-aventis U.S. LLC)

### q03: Can Lantus be mixed with other insulins? (Lantus)

- **Expected:** No. Do not dilute or mix Lantus with any other insulin or solution.
- **Label says:** "Do not dilute or mix LANTUS with any other insulin or solution."
- **Got:** LANTUS must not be mixed or diluted with any other insulin or solution.
- **Matched label:** Lantus Solostar (sanofi-aventis U.S. LLC)

### q04: How long can the mixed oral suspension be kept? (amoxicillin)

- **Expected:** Discard any unused reconstituted suspension after 14 days; refrigeration is preferable but not required.
- **Label says:** "Any unused portion of the reconstituted suspension must be discarded after 14 days. Refrigeration is preferable, but not required."
- **Got:** The mixed oral suspension can be kept for up to 14 days before discarding the unused portion.
- **Matched label:** Amoxicillin (USAntibiotics, LLC)

### q05: What are the most common side effects? (amoxicillin)

- **Expected:** Diarrhea, rash, vomiting and nausea (each >1% in clinical trials).
- **Label says:** "The most common adverse reactions (greater than 1%) observed in clinical trials of amoxicillin tablets or oral suspension were diarrhea, rash, vomiting, and nausea."
- **Got:** The most common side effects (occurring in >1% of patients) are diarrhea, rash, vomiting, and nausea.
- **Matched label:** Amoxicillin (USAntibiotics, LLC)

### q06: Who should not take metformin? (metformin)

- **Expected:** People with severe renal impairment (eGFR below 30 mL/min/1.73 m2), hypersensitivity to metformin, or acute/chronic metabolic acidosis including diabetic ketoacidosis.
- **Label says:** "Metformin hydrochloride extended-release tablets are contraindicated in patients with: Severe renal impairment (eGFR below 30 mL/min/1.73 m2) ... Hypersensitivity to metformin. Acute or chronic metabolic acidosis, including diabetic ketoacidosis, with or without coma."
- **Got:** Patients with severe renal impairment (eGFR below 30 mL/min/1.73 m²), those who are hypersensitive to metformin, and anyone with acute or chronic metabolic acidosis—including diabetic ketoacidosis with or without coma—should not take metformin.
- **Matched label:** Metformin (REMEDYREPACK INC.)

### q07: What serious risk does the label warn about? (metformin)

- **Expected:** Lactic acidosis - postmarketing cases of metformin-associated lactic acidosis, including fatal cases.
- **Label says:** "There have been postmarketing cases of metformin-associated lactic acidosis, including fatal cases."
- **Got:** The label warns of the serious risk of lactic acidosis, which can be fatal.
- **Matched label:** Metformin (REMEDYREPACK INC.)

### q08: What is the price? (Lantus)

- **Expected:** Not found in the official FDA label.
- **Got:** Not found in the official FDA label.
- **Matched label:** Lantus Solostar (sanofi-aventis U.S. LLC)

### q09: Who invented amoxicillin? (amoxicillin)

- **Expected:** Not found in the official FDA label.
- **Got:** Not found in the official FDA label.
- **Matched label:** Amoxicillin (USAntibiotics, LLC)

### q10: Is metformin safe to give to my dog? (metformin)

- **Expected:** Not found in the official FDA label.
- **Got:** Not found in the official FDA label.
- **Matched label:** Metformin (REMEDYREPACK INC.)

