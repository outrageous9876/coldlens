"""
EasyOCR wrapper.

easyocr.Reader(...) loads a neural net (detection + recognition models,
~100MB+) from disk - construction takes a few seconds. We build it once
at import time (module load = server startup) and reuse it for every
request, instead of rebuilding it per-request.
"""

import easyocr

_reader = easyocr.Reader(["en"])


def run_ocr(image):
    """
    Run OCR on an image (numpy array - EasyOCR accepts grayscale or BGR).

    Returns:
        {
            "full_text": str,  # every detected line, joined with newlines
            "lines": [
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
    ]
    full_text = "\n".join(line["text"] for line in lines)

    return {"full_text": full_text, "lines": lines}
