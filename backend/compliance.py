"""
Turns extracted label fields into compliance flags.
"""

from datetime import date, timedelta

from extract import LabelFields

EXPIRING_SOON_WINDOW_DAYS = 30
COLD_STORAGE_MAX_C = 8


def check_compliance(fields: LabelFields) -> dict:
    flags = {
        "expired": False,
        "expiring_soon": False,
        "cold_storage_required": False,
        "missing_batch_no": False,
    }

    if fields.expiry_date:
        expiry = date.fromisoformat(fields.expiry_date)
        today = date.today()
        if expiry < today:
            flags["expired"] = True
        elif expiry <= today + timedelta(days=EXPIRING_SOON_WINDOW_DAYS):
            flags["expiring_soon"] = True

    if fields.storage_temp_max is not None and fields.storage_temp_max <= COLD_STORAGE_MAX_C:
        flags["cold_storage_required"] = True

    if not fields.batch_no:
        flags["missing_batch_no"] = True

    return flags
