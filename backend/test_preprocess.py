"""
Quick manual test: run the preprocessing pipeline on one image and
dump every step's output as a PNG so you can eyeball before/after.

Usage:
    python test_preprocess.py <path-to-image>
"""

import sys
from pathlib import Path

import cv2

from preprocess import preprocess, save_steps


def main():
    if len(sys.argv) != 2:
        print("Usage: python test_preprocess.py <path-to-image>")
        sys.exit(1)

    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        sys.exit(1)

    _, steps = preprocess(image)

    out_dir = Path(__file__).resolve().parent.parent / "data" / "debug" / "manual"
    saved = save_steps(steps, out_dir)

    print(f"Processed {len(steps) - 1} steps for {image_path}:")
    for name, path in saved.items():
        print(f"  {name:<22} -> {path}")


if __name__ == "__main__":
    main()
