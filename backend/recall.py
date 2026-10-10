"""
Recall check via openFDA's drug enforcement endpoint.

Every FDA drug recall gets an "enforcement report": status (Ongoing /
Completed / Terminated), classification (Class I = most serious, can
cause serious harm or death; Class III = least), the reason, and a
free-text "code_info" field that usually lists the affected lot numbers,
e.g. "Lot #JKU4639A, Exp 10/2022". We search by product name and flag
BATCH MATCH when the scanned batch number appears as one of those lots.

Like the label endpoint, this only covers US-marketed products.
"""

import re

from fda import clean_product_name, query

ENFORCEMENT_URL = "https://api.fda.gov/drug/enforcement.json"
MIN_BATCH_LENGTH = 4  # shorter "batches" would match random numbers in code_info


def _normalize(text: str) -> str:
    """Upper-case and drop separators, so "CTZ-1173" matches "CTZ 1173"."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _lot_tokens(code_info: str) -> set[str]:
    return {_normalize(token) for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*", code_info)}


def _iso(yyyymmdd: str | None) -> str | None:
    if not yyyymmdd or len(yyyymmdd) != 8:
        return yyyymmdd
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:]}"


def check_recalls(product: str, batch: str | None = None) -> dict:
    term = clean_product_name(product)
    results = query(ENFORCEMENT_URL, f'product_description:"{term}"', limit=1000) if term else []
    batch_norm = _normalize(batch) if batch else ""

    recalls = []
    for r in results:
        code_info = r.get("code_info", "")
        recalls.append({
            "recall_number": r.get("recall_number"),
            "status": r.get("status"),
            "classification": r.get("classification"),
            "reason": r.get("reason_for_recall"),
            "date": _iso(r.get("recall_initiation_date")),
            "lots": code_info,  # raw lot text as published by FDA
            "product_description": r.get("product_description"),
            "recalling_firm": r.get("recalling_firm"),
            "batch_match": len(batch_norm) >= MIN_BATCH_LENGTH and batch_norm in _lot_tokens(code_info),
        })
    # batch matches first, then newest first
    recalls.sort(key=lambda rc: (rc["batch_match"], rc["date"] or ""), reverse=True)

    if any(rc["batch_match"] for rc in recalls):
        status = "batch_recalled"
    elif recalls:
        status = "recalls_exist"
    else:
        status = "no_recalls_found"

    return {
        "query": term,
        "batch": batch,
        "status": status,
        "recall_count": len(recalls),
        "recalls": recalls,
    }
