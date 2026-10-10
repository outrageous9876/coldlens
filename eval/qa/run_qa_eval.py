"""
Q&A eval: runs eval/qa/questions.json through POST /ask and grades each answer.

  answer correct  - in-label questions: found=True and every key-fact group
                    appears in the answer (each group = list of acceptable
                    phrasings, matched case-insensitively)
  right section   - at least one citation is from an expected section
  not found       - out-of-label questions: found=False

Only the QA_MODEL quota is used; any call to another model aborts the run.

Usage:
  python run_qa_eval.py --dry-run   # prints Groq budget, no calls
  python run_qa_eval.py             # writes eval/RESULTS_QA.md
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "backend"))

import extract  # noqa: E402
import rag  # noqa: E402
from fda import get_label  # noqa: E402

QUESTIONS = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
RESULTS_MD = HERE.parent / "RESULTS_QA.md"

tokens_used = 0
_original_call = rag._call_groq_with_retry


def _qa_only_call(**kwargs):
    global tokens_used
    if kwargs["model"] != rag.QA_MODEL:
        raise SystemExit(f"Refusing Groq call to {kwargs['model']} - QA eval may only use {rag.QA_MODEL}")
    response = _original_call(**kwargs)
    tokens_used += response.usage.total_tokens
    return response


def _no_extraction_call(**kwargs):
    raise SystemExit("Refusing extraction-model Groq call during QA eval")


rag._call_groq_with_retry = _qa_only_call
extract._call_groq_with_retry = _no_extraction_call


def normalize(text: str) -> str:
    # unify the many dash characters LLMs emit (‑ – — ‐) and collapse spaces
    text = re.sub(r"[‐-―−]", "-", text.lower())
    return re.sub(r"\s+", " ", text)


def grade(q: dict, result: dict) -> dict:
    answer_text = normalize(result["answer"])
    cited_sections = sorted({c["section"] for c in result["citations"]})
    if q["in_label"]:
        missing = [group[0] for group in q["key_facts"]
                   if not any(normalize(alt) in answer_text for alt in group)]
        return {
            "correct": result["found"] and not missing,
            "missing": missing,
            "section_ok": any(s in q["expected_sections"] for s in cited_sections),
            "cited": cited_sections,
        }
    return {"correct": not result["found"], "missing": [], "section_ok": None, "cited": cited_sections}


def main():
    from fastapi.testclient import TestClient
    from main import app

    labels = {p: get_label(p) for p in sorted({q["product"] for q in QUESTIONS})}
    uncached = [q for q in QUESTIONS if not rag.is_answer_cached(q["question"], labels[q["product"]])]
    print(f"GROQ BUDGET: {len(uncached)} fresh {rag.QA_MODEL} calls "
          f"(~1,050 tokens each, ~{len(uncached) * 1050} tokens); "
          f"{len(QUESTIONS) - len(uncached)} cached")
    if "--dry-run" in sys.argv:
        return

    client = TestClient(app)
    rows = []
    for q in QUESTIONS:
        response = client.post("/ask", json={"product_name": q["product"], "question": q["question"]})
        response.raise_for_status()
        result = response.json()
        rows.append({**q, "result": result, **grade(q, result)})
        print(f"{q['id']} {'OK ' if rows[-1]['correct'] else 'BAD'} {q['question']}", flush=True)

    write_markdown(rows)
    print(f"\nDone. {tokens_used} {rag.QA_MODEL} tokens. Wrote {RESULTS_MD}")


def write_markdown(rows):
    in_label = [r for r in rows if r["in_label"]]
    out_label = [r for r in rows if not r["in_label"]]
    yes_no = {True: "yes", False: "**NO**", None: "-"}

    lines = ["# ColdLens Label Q&A Evaluation\n",
             f"{len(rows)} questions over the Lantus, amoxicillin and metformin openFDA labels, "
             f"asked through `POST /ask` (model `{rag.QA_MODEL}`, top-4 retrieved chunks). "
             f"Questions, expected answers and source quotes: `eval/qa/questions.json`. "
             f"Groq tokens this run: {tokens_used} (cached answers cost 0).\n",
             "## Summary\n",
             "| Metric | Score |", "|---|---|",
             f"| Answer correct (in-label) | {sum(r['correct'] for r in in_label)}/{len(in_label)} |",
             f"| Right section cited (in-label) | {sum(bool(r['section_ok']) for r in in_label)}/{len(in_label)} |",
             f"| Correctly said \"not found\" (out-of-label) | {sum(r['correct'] for r in out_label)}/{len(out_label)} |",
             "\nAnswer correct = said found AND every key fact from the label appears in the "
             "answer (acceptable phrasings per fact in questions.json).\n",
             "## Per question\n",
             "| ID | Product | Question | Correct? | Right section? | Cited | Missing key facts |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['id']} | {r['product']} | {r['question']} | {yes_no[r['correct']]} | "
                     f"{yes_no[r['section_ok']]} | {', '.join(r['cited']) or '-'} | "
                     f"{', '.join(r['missing']) or '-'} |")

    lines.append("\n## Answers vs. expected\n")
    for r in rows:
        lines.append(f"### {r['id']}: {r['question']} ({r['product']})\n")
        lines.append(f"- **Expected:** {r['expected_answer']}")
        if r["source_quote"]:
            lines.append(f"- **Label says:** \"{r['source_quote']}\"")
        lines.append(f"- **Got:** {r['result']['answer']}")
        lines.append(f"- **Matched label:** {r['result']['matched_label']['brand_name']} "
                     f"({r['result']['matched_label']['manufacturer']})\n")

    RESULTS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
