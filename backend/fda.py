"""
openFDA drug label lookup.

openFDA (https://open.fda.gov) is the FDA's free public API. The
/drug/label endpoint serves the official prescribing information
("package insert") for US drugs, split into sections like
storage_and_handling or warnings. No API key needed: the free limits are
240 requests/minute and 1000/day per IP, which is plenty with the disk
cache below.

Label sections are not uniform across drugs:
  - newer labels use "warnings_and_cautions" instead of "warnings"
  - repackager labels often put storage info only under "how_supplied"
so each requested section has fallback field names, and the result
records which openFDA field each section actually came from.
"""

import hashlib
import json
import re
from pathlib import Path

import httpx

API_URL = "https://api.fda.gov/drug/label.json"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache" / "fda"

# requested section -> openFDA fields to try, in order
SECTION_FIELDS = {
    "storage_and_handling": ["storage_and_handling", "how_supplied"],
    "warnings": ["warnings", "warnings_and_cautions", "boxed_warning"],
    "dosage_and_administration": ["dosage_and_administration"],
    "contraindications": ["contraindications"],
    "adverse_reactions": ["adverse_reactions"],
}

# "Amoxicillin 500mg Capsules" -> "Amoxicillin"
_DOSE_RE = re.compile(r"\b\d+(\.\d+)?\s*(mg|mcg|µg|g|ml|iu|units?|%)(\s*/\s*\w+)?\b", re.I)
_FORM_RE = re.compile(r"\b(tablets?|capsules?|injection|solution|suspension|syrup|"
                      r"cream|ointment|vials?|pens?|oral|for)\b", re.I)


def clean_product_name(name: str) -> str:
    name = _DOSE_RE.sub(" ", name)
    name = _FORM_RE.sub(" ", name)
    return re.sub(r"\s+", " ", name).strip()


def query(url: str, search: str, limit: int = 20) -> list[dict]:
    """One openFDA query (any endpoint), cached to disk by URL + search."""
    cache_key = f"{url}|{search.lower()}|{limit}"
    cache_path = CACHE_DIR / f"{hashlib.sha256(cache_key.encode()).hexdigest()[:20]}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    response = httpx.get(url, params={"search": search, "limit": limit}, timeout=20)
    if response.status_code == 404:  # openFDA's way of saying "no matches"
        results = []
    else:
        response.raise_for_status()
        results = response.json().get("results", [])

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(results), encoding="utf-8")
    return results


def _search(field: str, term: str) -> list[dict]:
    return query(API_URL, f'openfda.{field}:"{term}"')


def _score(label: dict, term: str) -> tuple:
    """Higher is better: exact brand/generic name match, then how many
    requested sections exist natively, then most recent label version."""
    openfda = label.get("openfda", {})
    names = [n.lower() for n in openfda.get("brand_name", []) + openfda.get("generic_name", [])]
    exact = term.lower() in names
    native = sum(1 for section in SECTION_FIELDS if section in label)
    found = sum(1 for fields in SECTION_FIELDS.values() if any(f in label for f in fields))
    return (exact, found, native, label.get("effective_time", ""))


def _to_result(label: dict) -> dict:
    openfda = label.get("openfda", {})
    sections, sources = {}, {}
    for section, fields in SECTION_FIELDS.items():
        for field in fields:
            if field in label:
                # openFDA stores each section as a list of strings
                sections[section] = "\n".join(label[field]).strip()
                sources[section] = field
                break
    return {
        "brand_name": (openfda.get("brand_name") or [None])[0],
        "generic_name": (openfda.get("generic_name") or [None])[0],
        "manufacturer": (openfda.get("manufacturer_name") or [None])[0],
        "set_id": label.get("set_id"),
        "sections": sections,
        "section_sources": sources,  # which openFDA field each section came from
    }


def get_label(product: str) -> dict | None:
    """Best-matching openFDA label for a product name, or None."""
    term = clean_product_name(product)
    if not term:
        return None
    for field in ("brand_name", "generic_name"):
        results = _search(field, term)
        if results:
            best = max(results, key=lambda label: _score(label, term))
            return {"query": term, "matched_on": field, **_to_result(best)}
    return None
