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


def run_ocr(image, min_confidence=MIN_CONFIDENCE, group_rows=False):
    """
    Run OCR on an image (numpy array - EasyOCR accepts grayscale or BGR).

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
        }
    """
    results = _reader.readtext(image)

    lines = [
        {
            "text": text,
            "confidence": round(float(confidence), 4),
            "bbox": [[int(x), int(y)] for x, y in box],
        }
        for box, text, confidence in results
        if confidence >= min_confidence
    ]

    # group_rows=False keeps EasyOCR's raw detection order, one box per line.
    rows = _group_into_rows(lines) if group_rows else [[line] for line in lines]
    ordered_lines = [line for row in rows for line in row]
    full_text = "\n".join(" ".join(line["text"] for line in row) for row in rows)

    return {"full_text": full_text, "lines": ordered_lines}
