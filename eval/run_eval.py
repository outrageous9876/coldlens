"""
Runs the full ColdLens pipeline over the synthetic dataset in eval/data/
under 4 preprocessing variants, compares extracted fields to ground
truth, and reports accuracy per field / per variant / per distortion type.

Variants reuse one preprocess() call's intermediate snapshots rather than
re-running OpenCV 4x per image - the pipeline order is
grayscale -> denoise -> perspective_correction -> deskew -> adaptive_threshold,
so each variant is just a cumulative point along that one pass:
  no_preprocessing   -> "original" snapshot (untouched image)
  grayscale_denoise  -> "denoise" snapshot
  full_no_threshold  -> "deskew" snapshot (every step except the threshold)
  full_pipeline      -> "adaptive_threshold" snapshot (everything)

Usage: python run_eval.py
Output: eval/results.csv (long format, one row per image/variant/field)
        eval/RESULTS.md  (human-readable summary)
"""

import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
from rapidfuzz import fuzz

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from extract import extract_fields, is_cached  # noqa: E402
from ocr import run_ocr  # noqa: E402
from preprocess import preprocess  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_CSV = Path(__file__).resolve().parent / "results.csv"
RESULTS_MD = Path(__file__).resolve().parent / "RESULTS.md"
BASELINE_CSV = Path(__file__).resolve().parent / "results_baseline.csv"
BEFORE_AFTER_VARIANT = "full_no_threshold"

FIELDS = ["product", "batch_no", "mfg_date", "expiry_date", "storage_temp_min", "storage_temp_max"]

VARIANTS = {
    "no_preprocessing": "original",
    "grayscale_denoise": "denoise",
    "full_no_threshold": "deskew",
    "full_pipeline": "adaptive_threshold",
}

# Groq free-tier pacing: measured live from this account for GROQ_MODEL
# (openai/gpt-oss-20b) at generation time: 1000 requests/day, 8000
# tokens/minute. A low-reasoning-effort extraction call runs ~700-1500
# tokens depending on how garbled the OCR text is, so 8s between calls
# keeps sustained throughput comfortably under the TPM cap even on the
# noisiest variants. extract_fields() also retries 429s with backoff as
# a safety net if pacing isn't quite enough.
CALL_DELAY_SECONDS = 8

# "product" is free-text, and EasyOCR routinely confuses digits/letters in
# the bold title font (e.g. "500mg" -> "50Omg") even with no distortion
# applied - a 1-character miss that strict exact-match scores as a total
# failure. Score it with rapidfuzz's Levenshtein ratio too so a near-miss
# isn't indistinguishable from a wrong product entirely.
FUZZY_PRODUCT_THRESHOLD = 85


def normalize_text(value):
    if value is None:
        return None
    collapsed = re.sub(r"\s+", " ", value).strip().lower()
    return collapsed or None


def normalize_batch(value):
    if value is None:
        return None
    stripped = re.sub(r"\s+", "", value).upper()
    return stripped or None


def dates_match(a, b):
    return a == b  # both already ISO strings or None at this point


def numbers_match(a, b):
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) < 0.5


def product_fuzzy_ratio(extracted, expected):
    a, b = normalize_text(extracted), normalize_text(expected)
    if a is None or b is None:
        return 100.0 if a is None and b is None else 0.0
    return fuzz.ratio(a, b)


def compare(extracted: dict, ground_truth: dict) -> dict:
    return {
        "product": normalize_text(extracted.get("product")) == normalize_text(ground_truth.get("product")),
        "batch_no": normalize_batch(extracted.get("batch_no")) == normalize_batch(ground_truth.get("batch_no")),
        "mfg_date": dates_match(extracted.get("mfg_date"), ground_truth.get("mfg_date")),
        "expiry_date": dates_match(extracted.get("expiry_date"), ground_truth.get("expiry_date")),
        "storage_temp_min": numbers_match(extracted.get("storage_temp_min"), ground_truth.get("storage_temp_min")),
        "storage_temp_max": numbers_match(extracted.get("storage_temp_max"), ground_truth.get("storage_temp_max")),
    }


def load_samples():
    samples = []
    for json_path in sorted(DATA_DIR.glob("label_*.json")):
        img_path = json_path.with_suffix(".jpg")
        if not img_path.exists():
            continue
        ground_truth = json.loads(json_path.read_text())
        samples.append((json_path.stem, img_path, ground_truth))
    return samples


def load_baseline(path):
    """Load a previously-saved results.csv (copied aside as results_baseline.csv
    before a pipeline/prompt change) for the before/after comparison table."""
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        rows = []
        for r in csv.DictReader(f):
            rows.append({
                "image_id": r["image_id"],
                "variant": r["variant"],
                "distortion_type": r["distortion_type"],
                "distortion_severity": int(r["distortion_severity"]),
                "field": r["field"],
                "extracted": r["extracted"],
                "expected": r["expected"],
                "match": r["match"] == "True",
                "match_fuzzy": r["match_fuzzy"] == "True",
                "fuzzy_ratio": float(r["fuzzy_ratio"]) if r.get("fuzzy_ratio") else "",
            })
    return rows


def main():
    samples = load_samples()
    if not samples:
        print(f"No samples found in {DATA_DIR}. Run generate_synthetic.py first.")
        return

    max_calls = len(samples) * len(VARIANTS)
    print(f"{len(samples)} images x {len(VARIANTS)} variants = up to {max_calls} Groq API calls "
          f"(fewer in practice - variants that produce identical OCR text share a cache entry).")
    print(f"Pacing fresh (non-cached) calls {CALL_DELAY_SECONDS}s apart to stay under the account's "
          f"8000 tokens/minute limit. Worst case runtime: ~{max_calls * CALL_DELAY_SECONDS / 60:.0f} minutes.\n")

    rows = []
    api_calls_made = 0
    api_calls_cached = 0

    for i, (sample_id, img_path, ground_truth) in enumerate(samples, start=1):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[{i}/{len(samples)}] {sample_id}: could not load image, skipping")
            continue

        _, steps = preprocess(image)

        for variant_name, step_key in VARIANTS.items():
            variant_image = steps[step_key]
            ocr_result = run_ocr(variant_image)

            was_cached = is_cached(ocr_result["full_text"])
            fields = extract_fields(ocr_result["full_text"])
            if was_cached:
                api_calls_cached += 1
            else:
                api_calls_made += 1
                time.sleep(CALL_DELAY_SECONDS)

            extracted = fields.model_dump()
            matches = compare(extracted, ground_truth)
            product_ratio = product_fuzzy_ratio(extracted.get("product"), ground_truth.get("product"))

            for field in FIELDS:
                is_product = field == "product"
                rows.append({
                    "image_id": sample_id,
                    "variant": variant_name,
                    "distortion_type": ground_truth["distortion_type"],
                    "distortion_severity": ground_truth["distortion_severity"],
                    "field": field,
                    "extracted": extracted.get(field),
                    "expected": ground_truth.get(field),
                    "match": matches[field],
                    "match_fuzzy": (product_ratio >= FUZZY_PRODUCT_THRESHOLD) if is_product else matches[field],
                    "fuzzy_ratio": round(product_ratio, 1) if is_product else "",
                })

        print(f"[{i}/{len(samples)}] {sample_id} done "
              f"(distortion={ground_truth['distortion_type']} sev={ground_truth['distortion_severity']}) "
              f"- {api_calls_made} fresh calls, {api_calls_cached} cache hits so far")

    baseline_rows = load_baseline(BASELINE_CSV)

    write_csv(rows)
    write_markdown(rows, api_calls_made, api_calls_cached, baseline_rows)
    print(f"\nDone. {api_calls_made} fresh Groq calls, {api_calls_cached} served from cache.")
    print(f"Wrote {RESULTS_CSV} and {RESULTS_MD}")


def write_csv(rows):
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image_id", "variant", "distortion_type", "distortion_severity",
            "field", "extracted", "expected", "match", "match_fuzzy", "fuzzy_ratio",
        ])
        writer.writeheader()
        writer.writerows(rows)


def accuracy(rows, key="match"):
    if not rows:
        return 0.0
    return sum(1 for r in rows if r[key]) / len(rows)


def write_before_after(lines, baseline_rows, new_rows, variant=BEFORE_AFTER_VARIANT):
    if not baseline_rows:
        return

    lines.append(f"\n## Before vs. after improvements ({variant})\n")
    lines.append("Before = pre-fix ocr.py (raw EasyOCR detection order, no confidence filter) and "
                  "extract.py (no OCR character-confusion correction in the prompt). After = this "
                  "run - ocr.py now drops lines under 0.3 confidence and reorders text into rows "
                  "(top-to-bottom, left-to-right) before handing it to the LLM, and the prompt now "
                  "explicitly corrects O/0, S/5, l-I/1 confusions in numeric contexts.\n")
    lines.append("| Field | Before (strict) | After (strict) | Before (fuzzy*) | After (fuzzy*) |")
    lines.append("|---|---|---|---|---|")

    baseline_variant_rows = [r for r in baseline_rows if r["variant"] == variant]
    new_variant_rows = [r for r in new_rows if r["variant"] == variant]

    def field_acc(variant_rows, field, key):
        return accuracy([r for r in variant_rows if r["field"] == field], key)

    for field in FIELDS:
        b_strict = field_acc(baseline_variant_rows, field, "match")
        a_strict = field_acc(new_variant_rows, field, "match")
        b_fuzzy = field_acc(baseline_variant_rows, field, "match_fuzzy")
        a_fuzzy = field_acc(new_variant_rows, field, "match_fuzzy")
        lines.append(f"| {field} | {b_strict:.0%} | {a_strict:.0%} | {b_fuzzy:.0%} | {a_fuzzy:.0%} |")

    b_overall_strict = sum(field_acc(baseline_variant_rows, f, "match") for f in FIELDS) / len(FIELDS)
    a_overall_strict = sum(field_acc(new_variant_rows, f, "match") for f in FIELDS) / len(FIELDS)
    b_overall_fuzzy = sum(field_acc(baseline_variant_rows, f, "match_fuzzy") for f in FIELDS) / len(FIELDS)
    a_overall_fuzzy = sum(field_acc(new_variant_rows, f, "match_fuzzy") for f in FIELDS) / len(FIELDS)
    lines.append(f"| **Overall** | **{b_overall_strict:.0%}** | **{a_overall_strict:.0%}** | "
                  f"**{b_overall_fuzzy:.0%}** | **{a_overall_fuzzy:.0%}** |")
    lines.append("\n*fuzzy = strict match for every field except product, which uses "
                  f"rapidfuzz ratio >= {FUZZY_PRODUCT_THRESHOLD}.\n")


def write_markdown(rows, api_calls_made, api_calls_cached, baseline_rows=None):
    variants = list(VARIANTS.keys())
    distortion_types = sorted({r["distortion_type"] for r in rows})

    lines = []
    lines.append("# ColdLens Evaluation Results\n")
    lines.append(f"Synthetic dataset: {len(set(r['image_id'] for r in rows))} images "
                  f"x {len(variants)} preprocessing variants.")
    lines.append(f"Groq API usage this run: {api_calls_made} fresh calls, {api_calls_cached} cache hits.\n")

    write_before_after(lines, baseline_rows, rows)

    lines.append("\n## Overall accuracy per variant (strict exact match)\n")
    lines.append("Overall accuracy = mean of the 6 per-field accuracies (macro average), "
                  "not \"all 6 fields exactly right.\" Product name is particularly harsh under "
                  "strict matching - see the fuzzy-matching section below.\n")
    lines.append("| Variant | Overall | " + " | ".join(FIELDS) + " |")
    lines.append("|---" * (2 + len(FIELDS)) + "|")
    for variant in variants:
        variant_rows = [r for r in rows if r["variant"] == variant]
        per_field = [accuracy([r for r in variant_rows if r["field"] == f]) for f in FIELDS]
        overall = sum(per_field) / len(per_field)
        cells = " | ".join(f"{a:.0%}" for a in per_field)
        lines.append(f"| {variant} | **{overall:.0%}** | {cells} |")

    lines.append("\n## Product name: strict vs. fuzzy matching\n")
    lines.append("EasyOCR routinely confuses digits/letters in the bold title font (e.g. "
                  "\"500mg\" -> \"50Omg\"/\"SOOmg\"), even on undistorted images - a 1-character "
                  f"miss out of ~25 that strict exact-match scores as a total failure. Fuzzy match = "
                  f"`rapidfuzz.fuzz.ratio >= {FUZZY_PRODUCT_THRESHOLD}` on the case/whitespace-normalized strings.\n")
    lines.append("| Variant | Product (strict) | Product (fuzzy) | Overall (strict) | Overall (fuzzy product) |")
    lines.append("|---|---|---|---|---|")
    for variant in variants:
        variant_rows = [r for r in rows if r["variant"] == variant]
        product_rows = [r for r in variant_rows if r["field"] == "product"]
        strict_product = accuracy(product_rows, "match")
        fuzzy_product = accuracy(product_rows, "match_fuzzy")
        overall_strict = sum(accuracy([r for r in variant_rows if r["field"] == f], "match") for f in FIELDS) / len(FIELDS)
        overall_fuzzy = sum(accuracy([r for r in variant_rows if r["field"] == f], "match_fuzzy") for f in FIELDS) / len(FIELDS)
        lines.append(f"| {variant} | {strict_product:.0%} | {fuzzy_product:.0%} | "
                      f"{overall_strict:.0%} | {overall_fuzzy:.0%} |")

    near_misses = [r for r in rows if r["field"] == "product" and not r["match"] and r["match_fuzzy"]]
    if near_misses:
        lines.append("\n### Sample product near-misses (fails strict, passes fuzzy)\n")
        lines.append("| Image | Variant | Ground truth | Extracted | Fuzzy ratio |")
        lines.append("|---|---|---|---|---|")
        for r in near_misses[:5]:
            lines.append(f"| {r['image_id']} | {r['variant']} | {r['expected']} | {r['extracted']} | {r['fuzzy_ratio']} |")

    still_wrong = [r for r in rows if r["field"] == "product" and not r["match_fuzzy"]]
    if still_wrong:
        lines.append("\n### Sample product mismatches (fails even fuzzy)\n")
        lines.append("| Image | Variant | Ground truth | Extracted | Fuzzy ratio |")
        lines.append("|---|---|---|---|---|")
        for r in still_wrong[:5]:
            lines.append(f"| {r['image_id']} | {r['variant']} | {r['expected']} | {r['extracted']} | {r['fuzzy_ratio']} |")

    lines.append("\n## Accuracy by distortion type (overall, per variant)\n")
    lines.append("| Distortion | " + " | ".join(variants) + " |")
    lines.append("|---" * (1 + len(variants)) + "|")
    for dtype in distortion_types:
        cells = []
        for variant in variants:
            subset = [r for r in rows if r["variant"] == variant and r["distortion_type"] == dtype]
            n_images = len(set(r["image_id"] for r in subset))
            cells.append(f"{accuracy(subset):.0%} (n={n_images})")
        lines.append(f"| {dtype} | " + " | ".join(cells) + " |")

    RESULTS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
