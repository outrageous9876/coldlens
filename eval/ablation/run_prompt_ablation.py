"""
Prompt A/B on top of the restored OCR (old ocr.py behaviour, full_no_threshold):
  old_prompt     - baseline prompt (extract.SYSTEM_PROMPT as committed)
  narrow_prompt  - baseline prompt + O/0, S/5 correction for the product dose ONLY

Prompts are swapped by patching extract.SYSTEM_PROMPT / extract._PROMPT_HASH,
so each prompt gets its own cache entries.

Usage:
  python run_prompt_ablation.py --dry-run   # OCR only, prints Groq call budget
  python run_prompt_ablation.py             # writes PROMPT_ABLATION.md
"""

import hashlib
import sys
import time
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "backend"))

import extract  # noqa: E402
from ocr import run_ocr  # noqa: E402
from preprocess import preprocess  # noqa: E402
from run_eval import (BASELINE_CSV, BEFORE_AFTER_VARIANT, CALL_DELAY_SECONDS, FIELDS,  # noqa: E402
                      FUZZY_PRODUCT_THRESHOLD, compare, load_baseline, load_samples,
                      product_fuzzy_ratio)

NARROW_RULE = """- OCR often misreads digits in the dose inside the product name \
(e.g. "50Omg" or "SOOmg" is really "500mg"). In that dose ONLY, read O/o as 0 \
and S as 5. Do not apply this correction to any other field - leave batch \
numbers, dates and temperatures exactly as the rules below say."""

# The narrow rule is now the default prompt (extract.SYSTEM_PROMPT); the
# old prompt is that minus the rule.
NARROW_PROMPT = extract.SYSTEM_PROMPT
assert NARROW_RULE + "\n" in NARROW_PROMPT
OLD_PROMPT = NARROW_PROMPT.replace(NARROW_RULE + "\n", "", 1)

PROMPTS = {"old_prompt": OLD_PROMPT, "narrow_prompt": NARROW_PROMPT}

# Hard stop so this run can't eat the whole 200k tokens/day Groq quota:
# count real usage from every response and abort past MAX_RUN_TOKENS
# (finished results stay cached, so a rerun resumes where this stopped).
MAX_RUN_TOKENS = 140_000
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
    samples = load_samples()
    texts = {}
    for i, (sample_id, img_path, _) in enumerate(samples, start=1):
        _, steps = preprocess(cv2.imread(str(img_path)))
        texts[sample_id] = run_ocr(steps["deskew"])["full_text"]  # full_no_threshold
        print(f"OCR {i}/{len(samples)} {sample_id}", flush=True)

    budget = 0
    for name, prompt in PROMPTS.items():
        use_prompt(prompt)
        n = len({t for t in texts.values() if not extract.is_cached(t)})
        budget += n
        print(f"  {name}: {n} uncached unique texts")
    print(f"\nGROQ BUDGET: {budget} fresh calls (~{budget * CALL_DELAY_SECONDS / 60:.0f} min)")
    if "--dry-run" in sys.argv:
        return

    results, fresh = {}, 0
    for name, prompt in PROMPTS.items():
        use_prompt(prompt)
        rows = []
        for sample_id, _, gt in samples:
            text = texts[sample_id]
            was_cached = extract.is_cached(text)
            extracted = extract.extract_fields(text).model_dump()
            if not was_cached:
                fresh += 1
                time.sleep(CALL_DELAY_SECONDS)
            matches = compare(extracted, gt)
            ratio = product_fuzzy_ratio(extracted.get("product"), gt.get("product"))
            for f in FIELDS:
                rows.append({"field": f, "match": matches[f],
                             "match_fuzzy": ratio >= FUZZY_PRODUCT_THRESHOLD if f == "product" else matches[f]})
        results[name] = rows
        print(f"{name} done - {fresh} fresh calls, {tokens_used} tokens so far", flush=True)

    baseline = [r for r in load_baseline(BASELINE_CSV) if r["variant"] == BEFORE_AFTER_VARIANT]
    cols = {"baseline (committed run)": baseline, **results}

    def acc(rows, field, key="match"):
        sub = [r for r in rows if r["field"] == field]
        return sum(r[key] for r in sub) / len(sub)

    def overall(rows, key):
        return sum(acc(rows, f, key) for f in FIELDS) / len(FIELDS)

    lines = ["# Prompt ablation (restored OCR, full_no_threshold)\n",
             f"Fresh Groq calls this run: {fresh} ({tokens_used} tokens).\n",
             "| Field | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for f in FIELDS:
        lines.append(f"| {f} | " + " | ".join(f"{acc(r, f):.0%}" for r in cols.values()) + " |")
    lines.append("| product (fuzzy) | " + " | ".join(f"{acc(r, 'product', 'match_fuzzy'):.0%}" for r in cols.values()) + " |")
    lines.append("| **Overall (strict)** | " + " | ".join(f"**{overall(r, 'match'):.0%}**" for r in cols.values()) + " |")
    lines.append("| **Overall (fuzzy product)** | " + " | ".join(f"**{overall(r, 'match_fuzzy'):.0%}**" for r in cols.values()) + " |")
    lines.append(f"\nNarrow rule tested:\n\n```\n{NARROW_RULE}\n```")
    md = "\n".join(lines) + "\n"
    (HERE / "PROMPT_ABLATION.md").write_text(md, encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()
