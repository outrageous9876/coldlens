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
import logging
import os
import time
from datetime import date, datetime
from pathlib import Path

import groq
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "cache" / "extract_cache.json"
FAILURE_LOG_PATH = CACHE_PATH.parent / "extract_failures.jsonl"
MAX_RETRY_ATTEMPTS = 4

logger = logging.getLogger(__name__)

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
- OCR often misreads digits in the dose inside the product name \
(e.g. "50Omg" or "SOOmg" is really "500mg"). In that dose ONLY, read O/o as 0 \
and S as 5. Do not apply this correction to any other field - leave batch \
numbers, dates and temperatures exactly as the rules below say.
- Dates on labels come in many formats (MM/YYYY, MMM.YYYY, "EXP 12/26", \
DD/MM/YYYY, "20/6/2020", etc). Normalize all dates to ISO YYYY-MM-DD.
- Full numeric dates (day, month and year) written with slashes, dots or \
dashes - e.g. 08/11/2024, 20.01.2025, 02-09-2025 - are DAY first, then month, \
then year (Indian/European format): 08/11/2024 is 8 November 2024 = 2024-11-08. \
Only read such a date month-first if the label clearly uses US format, e.g. the \
second number is greater than 12 (03/25/2026).
- expiry_date must come ONLY from the date labeled EXP, Exp., Expiry, E.D., \
Use before or Use by. Never take it from a date labeled MFG, MFD, Mfd., \
Mfg. Dt., M.D. or Manufactured - that is the mfg_date.
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


_PROMPT_HASH = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:16]


def _cache_key(ocr_text: str) -> str:
    # Model and prompt are both part of the key so switching GROQ_MODEL or
    # editing SYSTEM_PROMPT can't silently serve stale results extracted
    # under the old model/prompt for unchanged OCR text.
    return hashlib.sha256(f"{GROQ_MODEL}:{_PROMPT_HASH}:{ocr_text}".encode("utf-8")).hexdigest()


def is_cached(ocr_text: str) -> bool:
    """Check whether extract_fields(ocr_text) would hit the cache, without
    making an API call. Lets callers that batch many calls (e.g. the eval
    harness) skip their rate-limit pacing delay on cache hits."""
    return _cache_key(ocr_text) in _load_cache()


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


def _log_failure(ocr_text, raw_reply, finish_reason, error):
    """Record why an extraction failed: a warning on stderr, plus one JSON
    line in extract_failures.jsonl with the model's raw reply, so failures
    can be diagnosed later without paying for another Groq call."""
    logger.warning("Extraction failed (%s: %s). finish_reason=%s raw reply: %r",
                   type(error).__name__, error, finish_reason, raw_reply)
    FAILURE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FAILURE_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "time": datetime.now().isoformat(timespec="seconds"),
            "model": GROQ_MODEL,
            "prompt_hash": _PROMPT_HASH,
            "error": f"{type(error).__name__}: {error}",
            "finish_reason": finish_reason,
            "raw_reply": raw_reply,
            "ocr_text": ocr_text,
        }) + "\n")


def extract_fields(ocr_text: str) -> LabelFields:
    cache = _load_cache()
    key = _cache_key(ocr_text)
    if key in cache:
        return LabelFields(**cache[key])

    extra_kwargs = {}
    if GROQ_MODEL.startswith("openai/gpt-oss"):
        # gpt-oss models spend a large, controllable number of hidden
        # "reasoning" tokens before answering - "low" is plenty for a
        # short structured-extraction task like this and roughly halves
        # token usage per call (measured: ~1330 -> ~685 tokens).
        extra_kwargs["reasoning_effort"] = "low"

    response = _call_groq_with_retry(
        model=GROQ_MODEL,
        response_format={"type": "json_object"},
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"OCR text:\n\n{ocr_text}"},
        ],
        **extra_kwargs,
    )

    raw_reply = response.choices[0].message.content
    try:
        data = json.loads(raw_reply)
        fields = LabelFields(**data)
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        # Model returned something we can't parse - treat as "nothing extracted"
        # rather than crashing the request. Not cached, so a retry can succeed.
        _log_failure(ocr_text, raw_reply, response.choices[0].finish_reason, error)
        return LabelFields()

    cache[key] = fields.model_dump()
    _save_cache(cache)

    return fields
