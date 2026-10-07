"""
Generates synthetic Indian-pharma-style label images with known ground
truth, for evaluating the OCR + extraction pipeline without needing a
pile of real photos.

Each sample is a (label_NNN.jpg, label_NNN.json) pair saved to eval/data/:
  - the .jpg is a rendered label photo, optionally distorted
  - the .json has the ground-truth field values plus which distortion
    (type + severity) was applied, normalized the same way extract.py's
    prompt is instructed to normalize dates (month-only date -> first day
    of month for mfg_date, last day of month for expiry_date).

Usage: python generate_synthetic.py [count]   (default count: 50)
"""

import calendar
import json
import random
import string
import sys
from datetime import date, timedelta
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).resolve().parent / "data"

FONT_DIR = Path("C:/Windows/Fonts")
BODY_FONTS = ["arial.ttf", "calibri.ttf", "tahoma.ttf", "verdana.ttf", "times.ttf", "courbd.ttf", "consola.ttf"]
BOLD_FONTS = ["arialbd.ttf", "calibrib.ttf", "tahomabd.ttf", "verdanab.ttf", "timesbd.ttf"]

BACKGROUND_COLORS = [
    (255, 255, 255), (235, 244, 251), (255, 249, 224),
    (234, 247, 234), (242, 242, 242), (255, 245, 238),
]
ACCENT_COLORS = [(0, 90, 160), (0, 120, 80), (180, 30, 40), (120, 80, 150), (200, 140, 0)]
TEXT_COLORS = [(20, 20, 20), (15, 25, 60), (40, 30, 25), (10, 40, 10)]

PRODUCTS = [
    "Paracetamol 500mg Tablets", "Azithromycin 250mg Tablets", "Cetirizine Syrup",
    "Amoxicillin 500mg Capsules", "Metformin 500mg Tablets", "ORS Sachet",
    "Insulin Glargine Injection", "COVAXIN Vaccine Vial", "Pantoprazole 40mg Tablets",
    "Ibuprofen 400mg Tablets", "Cough Syrup DX", "Ciprofloxacin 500mg Tablets",
    "Vitamin D3 60K IU Capsules", "ORS + Zinc Dispersible Tablets", "Ranitidine 150mg Tablets",
    "Doxycycline 100mg Capsules", "Diclofenac Gel", "Multivitamin Syrup",
]

BATCH_LABELS = ["B.No.", "Batch No.", "LOT", "B. No", "Batch No :"]
MFG_LABELS = ["Mfg. Date", "Mfg.Dt", "MFD", "Manufacturing Date", "Mfg Date"]
EXP_LABELS = ["Exp. Date", "EXP", "Use By", "Expiry Date", "Best Before"]

# (template string with {v} placeholder or None, storage_temp_min, storage_temp_max)
STORAGE_TEMPLATES = [
    ("Store below 25\u00b0C", None, 25),
    ("Store below 30\u00b0C. Protect from light.", None, 30),
    ("Store at 2\u00b0C-8\u00b0C. Do not freeze.", 2, 8),
    ("Store between 15\u00b0C-30\u00b0C", 15, 30),
    ("Store frozen at -20\u00b0C", -20, -20),
    (None, None, None),
]

# (strftime-ish formatter, has_day)
DATE_FORMATS = [
    (lambda d: d.strftime("%d/%m/%y"), True),
    (lambda d: d.strftime("%d/%m/%Y"), True),
    (lambda d: d.strftime("%m/%Y"), False),
    (lambda d: d.strftime("%b.%Y").upper(), False),
    (lambda d: d.strftime("%b %Y").upper(), False),
    (lambda d: "EXP " + d.strftime("%m/%y"), False),
]


def random_batch_no():
    style = random.choice(["digits", "letters_digits", "mixed"])
    if style == "digits":
        return "".join(random.choices(string.digits, k=random.randint(6, 7)))
    if style == "letters_digits":
        letters = "".join(random.choices(string.ascii_uppercase, k=random.randint(1, 2)))
        digits = "".join(random.choices(string.digits, k=random.randint(4, 5)))
        return letters + digits
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


def random_date(start, end):
    return start + timedelta(days=random.randint(0, (end - start).days))


def month_end(d):
    last_day = calendar.monthrange(d.year, d.month)[1]
    return date(d.year, d.month, last_day)


def build_date_field(real_date, is_expiry):
    """Pick a display format; compute the ground-truth ISO date using the
    same month-only normalization rule the Groq prompt is told to use."""
    formatter, has_day = random.choice(DATE_FORMATS)
    display = formatter(real_date)
    if has_day:
        iso = real_date.isoformat()
    else:
        iso = month_end(real_date).isoformat() if is_expiry else date(real_date.year, real_date.month, 1).isoformat()
    return display, iso


def build_label_content():
    product = random.choice(PRODUCTS)
    batch_no = random_batch_no()

    mfg_real = random_date(date(2022, 1, 1), date(2026, 6, 1))
    exp_real = mfg_real + timedelta(days=random.randint(365, 365 * 4))

    mfg_display, mfg_iso = build_date_field(mfg_real, is_expiry=False)
    exp_display, exp_iso = build_date_field(exp_real, is_expiry=True)

    storage_text, temp_min, temp_max = random.choice(STORAGE_TEMPLATES)

    ground_truth = {
        "product": product,
        "batch_no": batch_no,
        "mfg_date": mfg_iso,
        "expiry_date": exp_iso,
        "storage_temp_min": temp_min,
        "storage_temp_max": temp_max,
    }
    display = {
        "product": product,
        "batch_line": f"{random.choice(BATCH_LABELS)} {batch_no}",
        "mfg_line": f"{random.choice(MFG_LABELS)} : {mfg_display}",
        "exp_line": f"{random.choice(EXP_LABELS)} : {exp_display}",
        "storage_line": storage_text,
    }
    return ground_truth, display


def render_label(display):
    width, height = 900, 560
    bg = random.choice(BACKGROUND_COLORS)
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)

    accent = random.choice(ACCENT_COLORS)
    draw.rectangle([(0, 0), (width, 14)], fill=accent)

    text_color = random.choice(TEXT_COLORS)
    body_font_name = random.choice(BODY_FONTS)
    bold_font_name = random.choice(BOLD_FONTS)

    title_font = ImageFont.truetype(str(FONT_DIR / bold_font_name), random.randint(34, 42))
    body_size = random.randint(24, 30)
    body_font = ImageFont.truetype(str(FONT_DIR / body_font_name), body_size)

    x = random.randint(40, 70)
    y = random.randint(50, 80)
    draw.text((x, y), display["product"], font=title_font, fill=text_color)
    y += title_font.size + 35

    lines = [display["batch_line"], display["mfg_line"], display["exp_line"]]
    if display["storage_line"]:
        lines.append(display["storage_line"])

    for line in lines:
        draw.text((x, y), line, font=body_font, fill=text_color)
        y += body_size + random.randint(20, 30)

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def apply_blur(img, severity):
    k = {1: 3, 2: 7, 3: 13}[severity]
    return cv2.GaussianBlur(img, (k, k), 0)


def apply_rotation(img, severity):
    angle = {1: 3, 2: 8, 3: 15}[severity] * random.choice([-1, 1])
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def apply_perspective(img, severity):
    frac = {1: 0.02, 2: 0.05, 3: 0.09}[severity]
    h, w = img.shape[:2]
    jx, jy = w * frac, h * frac
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [random.uniform(0, jx), random.uniform(0, jy)],
        [w - random.uniform(0, jx), random.uniform(0, jy)],
        [w - random.uniform(0, jx), h - random.uniform(0, jy)],
        [random.uniform(0, jx), h - random.uniform(0, jy)],
    ])
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def apply_glare(img, severity):
    h, w = img.shape[:2]
    cx, cy = random.randint(0, w), random.randint(0, h)
    radius = {1: 80, 2: 150, 3: 250}[severity]
    opacity = {1: 0.3, 2: 0.55, 3: 0.8}[severity]
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = (np.clip(1 - dist / radius, 0, 1) ** 2) * opacity
    mask = mask[..., None]
    blended = img.astype(np.float32) * (1 - mask) + 255 * mask
    return np.clip(blended, 0, 255).astype(np.uint8)


def apply_noise(img, severity):
    sigma = {1: 10, 2: 22, 3: 38}[severity]
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_low_light(img, severity):
    factor = {1: 0.75, 2: 0.55, 3: 0.35}[severity]
    noise = np.random.normal(0, 8 / factor, img.shape).astype(np.float32)
    darker = img.astype(np.float32) * factor + noise
    return np.clip(darker, 0, 255).astype(np.uint8)


DISTORTIONS = {
    "blur": apply_blur,
    "rotation": apply_rotation,
    "perspective": apply_perspective,
    "glare": apply_glare,
    "noise": apply_noise,
    "low_light": apply_low_light,
}


def main():
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    random.seed(42)
    np.random.seed(42)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("label_*.jpg"):
        old.unlink()
    for old in OUT_DIR.glob("label_*.json"):
        old.unlink()

    combos = [("none", 0)] + [(t, s) for t in DISTORTIONS for s in (1, 2, 3)]

    for i in range(count):
        dtype, severity = combos[i % len(combos)]
        sample_id = f"label_{i + 1:03d}"

        ground_truth, display = build_label_content()
        img = render_label(display)
        if dtype != "none":
            img = DISTORTIONS[dtype](img, severity)

        img_path = OUT_DIR / f"{sample_id}.jpg"
        cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])

        gt = dict(ground_truth)
        gt["distortion_type"] = dtype
        gt["distortion_severity"] = severity
        (OUT_DIR / f"{sample_id}.json").write_text(json.dumps(gt, indent=2))

        print(f"{sample_id}: {dtype} sev={severity}  product={ground_truth['product'][:30]!r}")

    print(f"\nGenerated {count} samples in {OUT_DIR}")


if __name__ == "__main__":
    main()
