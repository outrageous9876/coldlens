"""
Ablation: which part of the round-2 ocr.py change caused the regression?

All configs use the NEW prompt (extract.py as committed) and the
full_no_threshold preprocessing variant (what /analyze uses):
  a_old_ocr       - ocr.py from commit 1e78bc2 (git show), raw detection order
  b_rows_no_conf  - current ocr.py, row grouping ON, confidence filter OFF
  c_current       - current ocr.py, row grouping ON, confidence filter 0.3

Usage:
  python run_ablation.py --dry-run   # OCR only, prints exact Groq call budget
  python run_ablation.py             # runs extraction, writes ABLATION.md
"""

import csv
import sys
import time
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                     # eval/  -> run_eval
sys.path.insert(0, str(HERE.parent.parent / "backend"))  # backend modules
sys.path.insert(0, str(HERE))                            # ocr_old

import ocr_old  # noqa: E402
from extract import extract_fields, is_cached  # noqa: E402
from ocr import run_ocr  # noqa: E402
from preprocess import preprocess  # noqa: E402
from run_eval import (BASELINE_CSV, BEFORE_AFTER_VARIANT, CALL_DELAY_SECONDS, FIELDS,  # noqa: E402
                      FUZZY_PRODUCT_THRESHOLD, compare, load_baseline, load_samples,
                      product_fuzzy_ratio)

CONFIGS = {
    "a_old_ocr": lambda img: ocr_old.run_ocr(img),
    "b_rows_no_conf": lambda img: run_ocr(img, min_confidence=0.0, group_rows=True),
    "c_current": lambda img: run_ocr(img, min_confidence=0.3, group_rows=True),
}


def main():
    dry_run = "--dry-run" in sys.argv
    samples = load_samples()

    # Pass 1: OCR everything (no Groq) so the budget is known up front.
    texts = {}  # (sample_id, config) -> ocr text
    for i, (sample_id, img_path, _) in enumerate(samples, start=1):
        _, steps = preprocess(cv2.imread(str(img_path)))
        image = steps["deskew"]  # full_no_threshold
        for name, ocr_fn in CONFIGS.items():
            texts[(sample_id, name)] = ocr_fn(image)["full_text"]
        print(f"OCR {i}/{len(samples)} {sample_id}", flush=True)

    uncached = {t for t in texts.values() if not is_cached(t)}
    for name in CONFIGS:
        n = len({texts[(s, name)] for s, _, _ in samples} & uncached)
        print(f"  {name}: {n} uncached unique texts")
    print(f"\nGROQ BUDGET: {len(uncached)} fresh calls "
          f"(~{len(uncached) * CALL_DELAY_SECONDS / 60:.0f} min at {CALL_DELAY_SECONDS}s pacing)")
    if dry_run:
        return

    # Pass 2: extract + score.
    results = {name: [] for name in CONFIGS}
    fresh = 0
    for sample_id, _, gt in samples:
        for name in CONFIGS:
            text = texts[(sample_id, name)]
            was_cached = is_cached(text)
            extracted = extract_fields(text).model_dump()
            if not was_cached:
                fresh += 1
                time.sleep(CALL_DELAY_SECONDS)
            matches = compare(extracted, gt)
            ratio = product_fuzzy_ratio(extracted.get("product"), gt.get("product"))
            for f in FIELDS:
                results[name].append({
                    "image_id": sample_id, "config": name, "field": f,
                    "extracted": extracted.get(f), "expected": gt.get(f),
                    "match": matches[f],
                    "match_fuzzy": ratio >= FUZZY_PRODUCT_THRESHOLD if f == "product" else matches[f],
                })
        print(f"{sample_id} done - {fresh} fresh calls so far", flush=True)

    baseline = [r for r in load_baseline(BASELINE_CSV) if r["variant"] == BEFORE_AFTER_VARIANT]
    write_outputs(results, baseline, fresh)


def field_acc(rows, field, key):
    sub = [r for r in rows if r["field"] == field]
    return sum(r[key] for r in sub) / len(sub)


def overall(rows, key):
    return sum(field_acc(rows, f, key) for f in FIELDS) / len(FIELDS)


def write_outputs(results, baseline, fresh):
    with open(HERE / "ablation_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(next(iter(results.values()))[0].keys()))
        w.writeheader()
        for rows in results.values():
            w.writerows(rows)

    cols = {"baseline": baseline, **results}
    lines = ["# OCR ablation (full_no_threshold, new prompt)\n",
             f"Fresh Groq calls this run: {fresh}.\n",
             "baseline = old ocr.py + old prompt (results_baseline.csv).\n",
             "| Field | " + " | ".join(cols) + " |",
             "|---" * (len(cols) + 1) + "|"]
    for field in FIELDS:
        lines.append(f"| {field} | " + " | ".join(f"{field_acc(r, field, 'match'):.0%}" for r in cols.values()) + " |")
    lines.append("| **Overall (strict)** | " + " | ".join(f"**{overall(r, 'match'):.0%}**" for r in cols.values()) + " |")
    lines.append("| **Overall (fuzzy product)** | " + " | ".join(f"**{overall(r, 'match_fuzzy'):.0%}**" for r in cols.values()) + " |")
    md = "\n".join(lines) + "\n"
    (HERE / "ABLATION.md").write_text(md, encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()
