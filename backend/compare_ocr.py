"""
Compares OCR output between the adaptive-thresholded image (end of the
preprocessing pipeline) and the plain grayscale image, for each photo in
data/raw/. Manual comparison tool - not part of the API.

Usage: python compare_ocr.py
"""

from pathlib import Path

import cv2

from ocr import run_ocr
from preprocess import preprocess

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def summarize(label, ocr_result):
    lines = ocr_result["lines"]
    if not lines:
        print(f"  [{label}] 0 lines detected")
        return
    avg_conf = sum(l["confidence"] for l in lines) / len(lines)
    print(f"  [{label}] {len(lines)} lines, avg confidence {avg_conf:.3f}")
    for l in lines:
        print(f"      ({l['confidence']:.2f}) {l['text']!r}")


def main():
    photos = sorted(p for p in RAW_DIR.iterdir() if p.is_file())
    for photo in photos:
        print(f"=== {photo.name} ===")
        image = cv2.imread(str(photo))
        if image is None:
            print("  could not load")
            continue

        final, steps = preprocess(image)
        grayscale = steps["grayscale"]

        summarize("adaptive_threshold", run_ocr(final))
        summarize("grayscale", run_ocr(grayscale))
        print()


if __name__ == "__main__":
    main()
