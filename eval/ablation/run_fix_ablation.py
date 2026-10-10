"""
Held-out fixes, tested one at a time and together, on BOTH eval sets
(main eval/data and eval/heldout), default /analyze pipeline only:
  before   - current prompt, current OCR
  date     - + day-first date rule and "expiry only from EXP-type line"
  rotation - + rotation fallback (rotated OCR retry only on low-confidence images)
  both     - both fixes

Usage:
  python run_fix_ablation.py --dry-run   # OCR only, prints Groq budget
  python run_fix_ablation.py             # writes FIX_ABLATION.md
"""

import hashlib
import sys
import time
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
EVAL_DIR = HERE.parent
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(EVAL_DIR.parent / "backend"))

import extract  # noqa: E402
from compliance import check_compliance  # noqa: E402
from main import ANALYZE_STEPS  # noqa: E402
from ocr import run_ocr  # noqa: E402
from preprocess import preprocess  # noqa: E402
from run_eval import (CALL_DELAY_SECONDS, FIELDS, FUZZY_PRODUCT_THRESHOLD, compare,  # noqa: E402
                      load_samples, product_fuzzy_ratio)

DATE_RULE = """- Full numeric dates (day, month and year) written with slashes, dots or \
dashes - e.g. 08/11/2024, 20.01.2025, 02-09-2025 - are DAY first, then month, \
then year (Indian/European format): 08/11/2024 is 8 November 2024 = 2024-11-08. \
Only read such a date month-first if the label clearly uses US format, e.g. the \
second number is greater than 12 (03/25/2026).
- expiry_date must come ONLY from the date labeled EXP, Exp., Expiry, E.D., \
Use before or Use by. Never take it from a date labeled MFG, MFD, Mfd., \
Mfg. Dt., M.D. or Manufactured - that is the mfg_date."""

# The date rule is now the default prompt (extract.SYSTEM_PROMPT); the
# "before" prompt is that minus the rule.
DATE_PROMPT = extract.SYSTEM_PROMPT
assert DATE_RULE + "\n" in DATE_PROMPT
BEFORE_PROMPT = DATE_PROMPT.replace(DATE_RULE + "\n", "", 1)


CONFIGS = {  # name -> (prompt, rotation fallback on?)
    "before": (BEFORE_PROMPT, False),
    "date": (DATE_PROMPT, False),
    "rotation": (BEFORE_PROMPT, True),
    "both": (DATE_PROMPT, True),
}
DATASETS = {"main": EVAL_DIR / "data", "heldout": EVAL_DIR / "heldout"}

# Leave >= 60k of today's 200k Groq tokens unused (72k already spent today).
MAX_RUN_TOKENS = 60_000
tokens_used = 0
_original_call = extract._call_groq_with_retry


def _counting_call(**kwargs):
    global tokens_used
    if tokens_used >= MAX_RUN_TOKENS:
        raise SystemExit(f"Token cap reached ({tokens_used} >= {MAX_RUN_TOKENS}), stopping.")
    response = _original_call(**kwargs)
    tokens_used += response.usage.total_tokens
    return response


extract._call_groq_with_retry = _counting_call


def use_prompt(prompt):
    extract.SYSTEM_PROMPT = prompt
    extract._PROMPT_HASH = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def main():
    # Pass 1: OCR with and without rotation for every image (no Groq).
    texts = {}  # (dataset, sample_id, rotation?) -> text
    samples = {name: load_samples(path) for name, path in DATASETS.items()}
    for ds, ds_samples in samples.items():
        for sample_id, img_path, _ in ds_samples:
            image, _ = preprocess(cv2.imread(str(img_path)), enabled_steps=ANALYZE_STEPS)
            texts[(ds, sample_id, False)] = run_ocr(image, rotation_fallback=False)["full_text"]
            texts[(ds, sample_id, True)] = run_ocr(image, rotation_fallback=True)["full_text"]
            print(f"OCR {ds}/{sample_id}", flush=True)
        changed = sum(texts[(ds, s, False)] != texts[(ds, s, True)] for s, _, _ in ds_samples)
        print(f"  {ds}: rotation changed OCR text on {changed}/{len(ds_samples)} images")

    budget = set()
    for name, (prompt, rot) in CONFIGS.items():
        use_prompt(prompt)
        n = 0
        for ds, ds_samples in samples.items():
            for sample_id, _, _ in ds_samples:
                text = texts[(ds, sample_id, rot)]
                if not extract.is_cached(text):
                    budget.add((extract._PROMPT_HASH, text))
                    n += 1
        print(f"  {name}: {n} uncached")
    print(f"\nGROQ BUDGET: {len(budget)} fresh calls (~{len(budget) * 700} tokens at ~700/call; "
          f"hard cap {MAX_RUN_TOKENS})")
    if "--dry-run" in sys.argv:
        return

    # Pass 2: extract + score every config on both datasets.
    results = {}  # (config, dataset) -> {"rows": [...], "flags_ok": int, "n": int}
    for name, (prompt, rot) in CONFIGS.items():
        use_prompt(prompt)
        for ds, ds_samples in samples.items():
            rows, flags_ok = [], 0
            for sample_id, _, gt in ds_samples:
                text = texts[(ds, sample_id, rot)]
                was_cached = extract.is_cached(text)
                fields = extract.extract_fields(text)
                if not was_cached:
                    time.sleep(CALL_DELAY_SECONDS)
                extracted = fields.model_dump()
                matches = compare(extracted, gt)
                ratio = product_fuzzy_ratio(extracted.get("product"), gt.get("product"))
                for f in FIELDS:
                    rows.append({"field": f, "match": matches[f],
                                 "match_fuzzy": ratio >= FUZZY_PRODUCT_THRESHOLD if f == "product" else matches[f]})
                expected = check_compliance(extract.LabelFields(**{f: gt.get(f) for f in FIELDS}))
                flags_ok += expected == check_compliance(fields)
            results[(name, ds)] = {"rows": rows, "flags_ok": flags_ok, "n": len(ds_samples)}
            print(f"{name}/{ds} done - {tokens_used} tokens so far", flush=True)

    write_report(results)


def acc(rows, field, key="match"):
    sub = [r for r in rows if r["field"] == field]
    return sum(r[key] for r in sub) / len(sub)


def overall(rows, key="match"):
    return sum(acc(rows, f, key) for f in FIELDS) / len(FIELDS)


def write_report(results):
    lines = ["# Held-out fix ablation (default /analyze pipeline)\n",
             f"Groq tokens this run: {tokens_used}.\n",
             "date = day-first numeric dates + expiry only from EXP-type lines. "
             "rotation = retry OCR with rotation_info=[90, 180, 270] only when the first pass "
             "has < 3 lines or mean confidence < 0.5, keeping the more confident pass.\n"]
    for ds in DATASETS:
        lines += [f"## {ds} ({results[('before', ds)]['n']} images)\n",
                  "| Field | " + " | ".join(CONFIGS) + " |", "|---" * (len(CONFIGS) + 1) + "|"]
        for f in FIELDS:
            lines.append(f"| {f} | " + " | ".join(f"{acc(results[(c, ds)]['rows'], f):.0%}" for c in CONFIGS) + " |")
        lines.append("| **Overall (strict)** | " + " | ".join(
            f"**{overall(results[(c, ds)]['rows']):.0%}**" for c in CONFIGS) + " |")
        lines.append("| **Overall (fuzzy product)** | " + " | ".join(
            f"**{overall(results[(c, ds)]['rows'], 'match_fuzzy'):.0%}**" for c in CONFIGS) + " |")
        lines.append("| Compliance flags fully correct | " + " | ".join(
            f"{results[(c, ds)]['flags_ok']}/{results[(c, ds)]['n']}" for c in CONFIGS) + " |\n")
    lines.append(f"Date rule tested:\n\n```\n{DATE_RULE}\n```")
    md = "\n".join(lines) + "\n"
    (HERE / "FIX_ABLATION.md").write_text(md, encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()
