"""
OpenCV preprocessing pipeline for label photos.

Each step is a small, independent function: (image) -> image.
This lets the eval harness toggle steps on/off by name and compare
OCR accuracy with different subsets of the pipeline enabled.
"""

from pathlib import Path

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Collapse the 3-channel BGR photo to 1-channel intensity.
    OCR doesn't care about color, and every later step (threshold, denoise)
    expects a single channel to work on."""
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def denoise(image: np.ndarray) -> np.ndarray:
    """Remove sensor/compression noise while keeping text edges sharp.
    fastNlMeansDenoising compares small patches across the whole image
    (not just neighboring pixels) and averages similar ones - good at
    killing speckle noise without blurring letters into mush."""
    return cv2.fastNlMeansDenoising(image, h=10, templateWindowSize=7, searchWindowSize=21)


def deskew(image: np.ndarray) -> np.ndarray:
    """Straighten small rotations (phone wasn't held level).
    We binarize, take the (x, y) coordinates of every dark pixel, and fit
    the smallest rotated rectangle around them with minAreaRect - its angle
    tells us how far the text is tilted. We then rotate the whole image
    back by that angle with warpAffine."""
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))

    if coords.shape[0] < 50:
        # Not enough dark pixels to trust an angle estimate - leave as-is.
        return image

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.5:
        # Rotation this small isn't worth the resample blur.
        return image

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image, rotation_matrix, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Sort 4 points into top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]       # smallest x+y -> top-left
    rect[2] = pts[np.argmax(s)]       # largest x+y -> bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]    # smallest x-y -> top-right
    rect[3] = pts[np.argmax(diff)]    # largest x-y -> bottom-left
    return rect


def correct_perspective(image: np.ndarray) -> np.ndarray:
    """If the label's 4 corners are visible (shot at an angle, not flat),
    warp it to a flat rectangle. Canny finds edges, findContours groups
    them into shapes, and we keep the largest 4-cornered shape as "the
    label". If nothing clean is found, we skip rather than guess."""
    edges = cv2.Canny(image, 50, 150)
    edges = cv2.dilate(edges, None, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image

    largest = max(contours, key=cv2.contourArea)
    image_area = image.shape[0] * image.shape[1]
    if cv2.contourArea(largest) < 0.2 * image_area:
        # Too small to be confidently "the label" - skip rather than warp garbage.
        return image

    perimeter = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * perimeter, True)
    if len(approx) != 4:
        return image

    ordered = _order_corners(approx.reshape(4, 2).astype("float32"))
    (tl, tr, br, bl) = ordered

    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if width < 10 or height < 10:
        return image

    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype="float32",
    )
    transform = cv2.getPerspectiveTransform(ordered, destination)
    return cv2.warpPerspective(image, transform, (width, height))


def adaptive_threshold(image: np.ndarray) -> np.ndarray:
    """Convert to pure black/white, but pick the black/white cutoff locally
    per neighborhood instead of one global cutoff (cv2.threshold). This
    handles uneven lighting across a label (shadow on one side, glare on
    the other) much better than a single global threshold would."""
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=35, C=11,
    )


# Ordered pipeline. Name here is what the eval harness toggles on/off.
STEPS = [
    ("grayscale", to_grayscale),
    ("denoise", denoise),
    ("perspective_correction", correct_perspective),
    ("deskew", deskew),
    ("adaptive_threshold", adaptive_threshold),
]


def preprocess(image: np.ndarray, enabled_steps: list[str] | None = None):
    """
    Run the pipeline in order.

    enabled_steps: optional list of step names to actually apply (others
    are skipped, passing the image through unchanged) - lets the eval
    script compare OCR accuracy with different steps on/off.

    Returns (final_image, steps) where `steps` is an ordered dict mapping
    step name -> image snapshot *after* that step, plus "original".
    """
    if enabled_steps is None:
        enabled_steps = [name for name, _ in STEPS]

    steps = {"original": image}
    current = image
    for name, func in STEPS:
        if name in enabled_steps:
            current = func(current)
        steps[name] = current

    return current, steps


def save_steps(steps: dict, out_dir: Path) -> dict[str, str]:
    """Write every step's image to out_dir as PNGs. Returns {name: path}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    for name, step_image in steps.items():
        out_path = out_dir / f"{name}.png"
        cv2.imwrite(str(out_path), step_image)
        saved[name] = str(out_path)
    return saved
