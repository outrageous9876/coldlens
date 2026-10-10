"""
EasyOCR wrapper.

easyocr.Reader(...) loads a neural net (detection + recognition models,
~100MB+) from disk - construction takes a few seconds. We build it once
at import time (module load = server startup) and reuse it for every
request, instead of rebuilding it per-request.
"""

import easyocr

_reader = easyocr.Reader(["en"])

# eval/ablation/ABLATION.md: the 0.3 confidence filter cost ~6 points
# (it drops real but blurry date/temp text), and row grouping was neutral
# to slightly worse. Both are off by default; raw EasyOCR order wins.
MIN_CONFIDENCE = 0.0

# Rotation fallback: a sideways label still produces ~20 junk boxes, so
# line count alone doesn't catch it - low average confidence does. On the
# eval sets every normal image averages >= 0.68; sideways/badly shadowed
# ones fall around 0.4.
ROTATION_ANGLES = [90, 180, 270]
FALLBACK_MIN_LINES = 3
FALLBACK_MIN_MEAN_CONFIDENCE = 0.5


def _y_range(line):
    ys = [p[1] for p in line["bbox"]]
    return min(ys), max(ys)


def _x_min(line):
    return min(p[0] for p in line["bbox"])


def _group_into_rows(lines):
    """EasyOCR returns lines in roughly top-to-bottom detection order, but
    doesn't guarantee that - and never sorts left-to-right within a line
    of text. Group boxes into visual rows by vertical overlap (two boxes
    are "the same row" if one's y-range covers at least half the other's
    height), then sort each row left-to-right, so reading order matches
    how a human would actually read the label."""
    rows = []
    for line in sorted(lines, key=lambda l: _y_range(l)[0]):
        y0, y1 = _y_range(line)
        height = y1 - y0
        for row in rows:
            overlap = min(y1, row["y1"]) - max(y0, row["y0"])
            row_height = row["y1"] - row["y0"]
            if overlap > 0.5 * min(height, row_height):
                row["lines"].append(line)
                row["y0"] = min(row["y0"], y0)
                row["y1"] = max(row["y1"], y1)
                break
        else:
            rows.append({"y0": y0, "y1": y1, "lines": [line]})

    rows.sort(key=lambda r: r["y0"])
    for row in rows:
        row["lines"].sort(key=_x_min)
    return [row["lines"] for row in rows]


def _read(image, rotation_info=None):
    return [
        {
            "text": text,
            "confidence": round(float(confidence), 4),
            "bbox": [[int(x), int(y)] for x, y in box],
        }
        for box, text, confidence in _reader.readtext(image, rotation_info=rotation_info)
    ]


def _mean_confidence(lines):
    return sum(line["confidence"] for line in lines) / len(lines) if lines else 0.0


def run_ocr(image, min_confidence=MIN_CONFIDENCE, group_rows=False, rotation_fallback=False):
    """
    Run OCR on an image (numpy array - EasyOCR accepts grayscale or BGR).

    Rotation fallback is OFF by default until the held-out ablation
    (eval/ablation/run_fix_ablation.py) shows it helps. When on: normal
    OCR runs first. Only if it finds very little readable text
    (fewer than FALLBACK_MIN_LINES lines, or mean confidence below
    FALLBACK_MIN_MEAN_CONFIDENCE) and rotation_fallback is on, OCR runs
    again with EasyOCR's rotation_info (each box also tried at 90/180/270
    degrees - about 4x slower), and the more confident of the two passes
    is kept. So normal uploads stay fast and sideways labels get read.

    Optional (both off by default, see MIN_CONFIDENCE): drop lines below
    min_confidence, and with group_rows=True reorder boxes into visual rows
    (top-to-bottom, left-to-right). With defaults, full_text is every
    detected box in EasyOCR's order, one per line.

    Returns:
        {
            "full_text": str,  # rows joined with newlines (with defaults,
                                # one detected box per row)
            "lines": [          # same reading order as full_text
                {"text": str, "confidence": float, "bbox": [[x, y], ...]},
                ...
            ],
            "rotation_retry": bool,  # True if the rotated pass was used
        }
    """
    raw = _read(image)
    rotation_retry = False
    if rotation_fallback and (len(raw) < FALLBACK_MIN_LINES
                              or _mean_confidence(raw) < FALLBACK_MIN_MEAN_CONFIDENCE):
        rotated = _read(image, rotation_info=ROTATION_ANGLES)
        if _mean_confidence(rotated) > _mean_confidence(raw):
            raw, rotation_retry = rotated, True

    lines = [line for line in raw if line["confidence"] >= min_confidence]

    # group_rows=False keeps EasyOCR's raw detection order, one box per line.
    rows = _group_into_rows(lines) if group_rows else [[line] for line in lines]
    ordered_lines = [line for row in rows for line in row]
    full_text = "\n".join(" ".join(line["text"] for line in row) for row in rows)

    return {"full_text": full_text, "lines": ordered_lines, "rotation_retry": rotation_retry}
