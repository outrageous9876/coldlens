"""
Turns raw OCR text into structured label fields using a Groq-hosted LLM
in JSON mode.

Model choice: Groq retired every Llama chat model in 2026 (llama-3.1-8b-instant
shut down 2026-08-16, llama-4-scout shut down 2026-07-17) - openai/gpt-oss-20b
is Groq's own documented replacement for llama-3.1-8b-instant: a small,
fast open-weight model, plenty for a short structured-extraction prompt
like this. Overridable via the GROQ_MODEL env var.
"""

import hashlib
import json
import os
import time
from datetime import date
from pathlib import Path

import groq
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "cache" / "extract_cache.json"
MAX_RETRY_ATTEMPTS = 4

_client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You extract structured data from OCR text scanned off a \
medicine/vaccine/food label. The OCR text may contain misreads (e.g. O/0, \
I/1 confusion) - use your best judgement.

Return ONLY a JSON object with exactly these keys, no others:
  product (string or null) - the product/brand name
  batch_no (string or null) - batch number / lot number / lot no.
  mfg_date (string or null) - manufacturing date, ISO format YYYY-MM-DD
  expiry_date (string or null) - expiry date, ISO format YYYY-MM-DD
  storage_temp_min (number or null) - minimum storage temperature in Celsius
  storage_temp_max (number or null) - maximum storage temperature in Celsius

Rules:
- If a field is not present in the text, use null. Never guess or invent a value.
- Dates on labels come in many formats (MM/YYYY, MMM.YYYY, "EXP 12/26", \
DD/MM/YYYY, "20/6/2020", etc). Normalize all dates to ISO YYYY-MM-DD.
- If a date only gives month and year (no day), use the FIRST day of that \
month for mfg_date, and the LAST day of that month for expiry_date (labels \
with a month-only expiry are conventionally valid through the end of that month).
- Storage temperatures given in Fahrenheit must be converted to Celsius.
- A single-sided instruction like "Store below 25C" means storage_temp_max=25 \
and storage_temp_min=null. "Store frozen at -20C" means storage_temp_min and \
storage_temp_max both -20.
- Output must be valid JSON. No markdown, no commentary, no trailing text."""


class LabelFields(BaseModel):
    product: str | None = None
    batch_no: str | None = None
    mfg_date: str | None = None
    expiry_date: str | None = None
    storage_temp_min: float | None = None
    storage_temp_max: float | None = None

    @field_validator("mfg_date", "expiry_date")
    @classmethod
    def _must_be_iso_date(cls, value):
        """A malformed date is as good as missing - null it out rather than
        passing a string the compliance checks can't parse."""
        if value is None:
            return None
        try:
            date.fromisoformat(value)
        except ValueError:
            return None
        return value


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


def _cache_key(ocr_text: str) -> str:
    # Model is part of the key so switching GROQ_MODEL doesn't serve stale
    # results extracted by a different model.
    return hashlib.sha256(f"{GROQ_MODEL}:{ocr_text}".encode("utf-8")).hexdigest()


def _call_groq_with_retry(**kwargs):
    """Groq's SDK already retries transport errors internally, but we add
    our own backoff specifically for 429 (rate limit) since those are
    expected during bursty usage and worth a longer, visible retry."""
    backoff_seconds = 1.0
    for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
        try:
            return _client.chat.completions.create(**kwargs)
        except groq.RateLimitError:
            if attempt == MAX_RETRY_ATTEMPTS:
                raise
            time.sleep(backoff_seconds)
            backoff_seconds *= 2


def extract_fields(ocr_text: str) -> LabelFields:
    cache = _load_cache()
    key = _cache_key(ocr_text)
    if key in cache:
        return LabelFields(**cache[key])

    response = _call_groq_with_retry(
        model=GROQ_MODEL,
        response_format={"type": "json_object"},
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"OCR text:\n\n{ocr_text}"},
        ],
    )

    try:
        data = json.loads(response.choices[0].message.content)
        fields = LabelFields(**data)
    except (json.JSONDecodeError, TypeError, ValueError):
        # Model returned something we can't parse - treat as "nothing extracted"
        # rather than crashing the request. Not cached, so a retry can succeed.
        return LabelFields()

    cache[key] = fields.model_dump()
    _save_cache(cache)

    return fields
