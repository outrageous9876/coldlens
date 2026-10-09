"""Held-out test set generator for ColdLens (independent of eval/generate_synthetic.py).
Styles the original generator never produced: blister foil backs, inkjet dot-matrix
batch/expiry stamps, curved vial/bottle labels, carton end flaps, cast shadows,
specular glare, JPEG artefacts, table backgrounds."""
import json, os, random
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

random.seed(7); np.random.seed(7)
OUT = "/home/claude/heldout/real_like"
os.makedirs(OUT, exist_ok=True)
F = "/usr/share/fonts/truetype/"
FONTS = {
    "sans": F + "dejavu/DejaVuSans.ttf", "sansb": F + "dejavu/DejaVuSans-Bold.ttf",
    "cond": F + "dejavu/DejaVuSansCondensed.ttf", "condb": F + "dejavu/DejaVuSansCondensed-Bold.ttf",
    "serifb": F + "dejavu/DejaVuSerif-Bold.ttf", "mono": F + "dejavu/DejaVuSansMono.ttf",
    "lib": F + "liberation/LiberationSans-Regular.ttf", "libb": F + "liberation/LiberationSans-Bold.ttf",
}
def font(k, s): return ImageFont.truetype(FONTS[k], s)

# ---------- surfaces ----------
def foil(w, h):
    base = np.linspace(170, 215, w)[None, :].repeat(h, 0)
    brushed = cv2.GaussianBlur(np.random.normal(0, 18, (h, w)), (41, 1), 0)
    img = np.clip(base + brushed, 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([img, img, (img * 1.02).clip(0, 255).astype(np.uint8)]))

def paper(w, h, color=(246, 244, 238)):
    arr = np.full((h, w, 3), color, np.float32) + np.random.normal(0, 3, (h, w, 1))
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))

def table_bg(w, h):
    grain = cv2.GaussianBlur(np.random.normal(0, 25, (h, w)), (1, 61), 0)
    r = (120 + grain).clip(0, 255); g = (85 + grain * .8).clip(0, 255); b = (60 + grain * .6).clip(0, 255)
    return np.dstack([b, g, r]).astype(np.uint8)  # BGR

# ---------- inkjet dot-matrix text ----------
def inkjet(draw_img, xy, text, size, color=(25, 25, 30), pitch=3, dot=1.9):
    f = font("monob" if False else "mono", size)
    W, H = draw_img.size
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).text(xy, text, font=f, fill=255)
    m = np.array(mask)
    d = ImageDraw.Draw(draw_img)
    ys, xs = np.where(m[::pitch, ::pitch] > 120)
    for y, x in zip(ys * pitch, xs * pitch):
        jx, jy = np.random.uniform(-.6, .6, 2)
        d.ellipse([x + jx - dot, y + jy - dot, x + jx + dot, y + jy + dot], fill=color)

# ---------- photo effects (BGR numpy) ----------
def cylinder(img, strength=.35):
    h, w = img.shape[:2]
    xs = np.linspace(-1, 1, w)
    src_x = (np.arcsin(np.clip(xs * np.sin(strength * np.pi / 2) / np.sin(strength * np.pi / 2), -1, 1))
             / (np.pi / 2)) if False else np.sin(xs * strength * np.pi / 2) / np.sin(strength * np.pi / 2)
    map_x = ((src_x + 1) / 2 * (w - 1)).astype(np.float32)[None, :].repeat(h, 0)
    map_y = np.arange(h, dtype=np.float32)[:, None].repeat(w, 1)
    out = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    shade = (0.75 + 0.25 * np.cos(xs * np.pi / 2 * 1.1))[None, :, None]
    return (out * shade).clip(0, 255).astype(np.uint8)

def place(label, angle=0, persp=0.0, scale=0.75, W=1280, H=900):
    bg = table_bg(W, H)
    lh, lw = label.shape[:2]
    s = min(W * scale / lw, H * scale / lh)
    lw2, lh2 = int(lw * s), int(lh * s)
    lab = cv2.resize(label, (lw2, lh2), interpolation=cv2.INTER_AREA)
    cx, cy = W / 2 + random.uniform(-40, 40), H / 2 + random.uniform(-30, 30)
    src = np.float32([[0, 0], [lw2, 0], [lw2, lh2], [0, lh2]])
    pts = np.float32([[-lw2/2, -lh2/2], [lw2/2, -lh2/2], [lw2/2, lh2/2], [-lw2/2, lh2/2]])
    pts[0, 0] += persp * lw2; pts[3, 0] += persp * lw2 * .3   # one side recedes
    pts[0, 1] += persp * lh2 * .5; pts[3, 1] -= persp * lh2 * .2
    a = np.deg2rad(angle); R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    dst = (pts @ R.T + [cx, cy]).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(lab, M, (W, H))
    mask = cv2.warpPerspective(np.full((lh2, lw2), 255, np.uint8), M, (W, H))
    # soft drop shadow
    sh = cv2.GaussianBlur(np.roll(np.roll(mask, 14, 0), 10, 1), (41, 41), 0)[..., None] / 255.0
    bg = (bg * (1 - 0.45 * sh)).astype(np.uint8)
    m = (mask[..., None] / 255.0)
    return (warped * m + bg * (1 - m)).astype(np.uint8)

def light(img, glare=0.0, shadow=0.0, dark=1.0):
    h, w = img.shape[:2]
    out = img.astype(np.float32) * dark
    if shadow:
        yy, xx = np.mgrid[0:h, 0:w]
        out *= (1 - shadow * (xx / w > 0.55 + 0.1 * np.sin(yy / 90)))[..., None]
        out = cv2.GaussianBlur(out, (3, 3), 0)
    if glare:
        gx, gy = random.uniform(.3, .7) * w, random.uniform(.3, .6) * h
        yy, xx = np.mgrid[0:h, 0:w]
        blob = np.exp(-(((xx - gx) / (w * .12)) ** 2 + ((yy - gy) / (h * .07)) ** 2))
        out += (255 * glare * blob)[..., None]
    return out.clip(0, 255).astype(np.uint8)

def camera(img, blur=0, noise=4, q=72):
    if blur: img = cv2.GaussianBlur(img, (0, 0), blur)
    img = (img.astype(np.float32) + np.random.normal(0, noise, img.shape)).clip(0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)

def to_bgr(pil): return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# ---------- label templates ----------
def blister(p):
    im = foil(1100, 520); d = ImageDraw.Draw(im)
    for r in range(3):
        for c in range(4):
            d.ellipse([60 + c * 260, 40 + r * 150, 200 + c * 260, 150 + r * 150], outline=(150, 150, 155), width=3)
    for i in range(0, 1100, 275):  # repeated print, typical strip
        d.text((i + 20, 10), p["product"].split(" Tablets")[0], font=font("condb", 26), fill=(30, 60, 140))
    d.rectangle([0, 400, 1100, 520], fill=(200, 200, 205))
    inkjet(im, (40, 412), f"B.No. {p['batch']}   MFG. {p['mfg_s']}", 40, pitch=3)
    inkjet(im, (40, 462), f"EXP. {p['exp_s']}   {p['price']}", 40, pitch=3)
    d.text((870, 440), p["storage_s"], font=font("condb", 20), fill=(40, 40, 40))
    return im

def carton_flap(p):
    im = paper(1100, 460, (250, 250, 252)); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 1100, 70], fill=p["brand_color"])
    d.text((30, 12), p["product"], font=font("libb", 40), fill=(255, 255, 255))
    d.text((30, 95), p["storage_s"], font=font("lib", 26), fill=(30, 30, 30))
    d.text((30, 135), "Keep out of reach of children.", font=font("lib", 22), fill=(60, 60, 60))
    d.rectangle([30, 200, 620, 420], outline=(120, 120, 120), width=2)
    inkjet(im, (50, 215), f"BATCH NO.: {p['batch']}", 38)
    inkjet(im, (50, 280), f"MFD: {p['mfg_s']}", 38)
    inkjet(im, (50, 345), f"EXP: {p['exp_s']}", 38)
    d.text((700, 380), "Mfd. by: " + p["maker"], font=font("cond", 20), fill=(70, 70, 70))
    return im

def vial(p):
    im = paper(1300, 420, (255, 255, 255)); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 1300, 18], fill=p["brand_color"]); d.rectangle([0, 402, 1300, 420], fill=p["brand_color"])
    d.text((160, 45), p["product"], font=font("sansb", 46), fill=(20, 20, 20))
    d.text((160, 115), "For intramuscular use only. Single dose vial 0.5 mL", font=font("sans", 24), fill=(60, 60, 60))
    d.rectangle([160, 170, 760, 230], fill=(0, 90, 170))
    d.text((175, 182), p["storage_s"], font=font("sansb", 28), fill=(255, 255, 255))
    d.text((160, 260), f"Lot: {p['batch']}", font=font("sansb", 32), fill=(20, 20, 20))
    d.text((160, 305), f"Mfg: {p['mfg_s']}", font=font("sans", 30), fill=(20, 20, 20))
    d.text((160, 350), f"Exp: {p['exp_s']}", font=font("sans", 30), fill=(20, 20, 20))
    d.text((860, 300), p["maker"], font=font("cond", 22), fill=(80, 80, 80))
    return im

def bottle(p):
    im = paper(1300, 520, (255, 248, 225)); d = ImageDraw.Draw(im)
    d.ellipse([-200, -300, 600, 260], fill=p["brand_color"])
    d.text((80, 40), p["product"].split(" ")[0], font=font("serifb", 54), fill=(255, 255, 255))
    d.text((80, 300), p["product"], font=font("libb", 34), fill=(30, 30, 30))
    d.text((80, 350), p["storage_s"], font=font("lib", 26), fill=(30, 30, 30))
    d.text((80, 395), "Shake well before use.", font=font("lib", 24), fill=(80, 80, 80))
    d.text((760, 120), f"B. No.  {p['batch']}", font=font("monob" if False else "mono", 34), fill=(10, 10, 10))
    d.text((760, 175), f"Mfg. Dt. {p['mfg_s']}", font=font("mono", 34), fill=(10, 10, 10))
    d.text((760, 230), f"Exp. Dt. {p['exp_s']}", font=font("mono", 34), fill=(10, 10, 10))
    return im

def side_panel(p):
    im = paper(1200, 600, (238, 245, 250)); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 420, 600], fill=p["brand_color"])
    d.text((30, 60), p["product"].split(" ")[0], font=font("sansb", 44), fill=(255, 255, 255))
    d.text((30, 120), " ".join(p["product"].split(" ")[1:]), font=font("sans", 28), fill=(255, 255, 255))
    d.text((460, 40), "Composition: Each tablet contains", font=font("cond", 24), fill=(40, 40, 40))
    d.text((460, 75), p["product"].split(" Tablets")[0] + " ......", font=font("cond", 24), fill=(40, 40, 40))
    d.text((460, 140), "Dosage: As directed by the physician.", font=font("cond", 24), fill=(40, 40, 40))
    d.text((460, 190), p["storage_s"], font=font("condb", 26), fill=(160, 20, 20))
    d.text((460, 230), "Protect from light and moisture.", font=font("cond", 24), fill=(40, 40, 40))
    d.text((460, 320), f"B.No.: {p['batch']}", font=font("monob" if False else "mono", 32), fill=(15, 15, 15))
    d.text((460, 370), f"M.D.: {p['mfg_s']}", font=font("mono", 32), fill=(15, 15, 15))
    d.text((460, 420), f"E.D.: {p['exp_s']}", font=font("mono", 32), fill=(15, 15, 15))
    d.text((460, 520), "Marketed by " + p["maker"], font=font("cond", 20), fill=(90, 90, 90))
    return im

# ---------- the 10 samples (ground truth uses full dates: no month-only ambiguity) ----------
S = [
 dict(t=blister, product="Paracetamol 650 mg", batch="PCT24187", mfg="2025-03-12", exp="2028-02-28",
      mfg_s="12/03/2025", exp_s="28/02/2028", storage_s="Store below 30°C", tmin=None, tmax=30,
      price="M.R.P. Rs.32.10", fx=dict(angle=8, persp=.08, glare=.55, blur=.6)),
 dict(t=carton_flap, product="Azithromycin 500 mg Tablets", batch="AZM5L042", mfg="2025-06-04", exp="2027-05-31",
      mfg_s="04-JUN-2025", exp_s="31-MAY-2027", storage_s="Store below 25°C. Protect from light.", tmin=None, tmax=25,
      fx=dict(angle=-5, persp=.15, shadow=.35, blur=.4)),
 dict(t=vial, product="Hepatitis B Vaccine (rDNA)", batch="HBV2509A", mfg="2025-09-15", exp="2027-08-31",
      mfg_s="15/09/2025", exp_s="31/08/2027", storage_s="Store at 2°C to 8°C. Do not freeze.", tmin=2, tmax=8,
      fx=dict(angle=3, cyl=.45, glare=.35, blur=.5)),
 dict(t=bottle, product="Cetirizine Oral Solution", batch="CTZ-1173", mfg="2025-01-20", exp="2027-12-31",
      mfg_s="20.01.2025", exp_s="31.12.2027", storage_s="Store below 30°C.", tmin=None, tmax=30,
      fx=dict(angle=-12, cyl=.35, dark=.6, blur=.7)),
 dict(t=side_panel, product="Metformin 500 mg Tablets", batch="MTF0925B", mfg="2025-09-02", exp="2027-08-31",
      mfg_s="02/09/2025", exp_s="31/08/2027", storage_s="Store in a cool, dry place below 25°C", tmin=None, tmax=25,
      fx=dict(angle=17, persp=.05, blur=.3)),
 dict(t=blister, product="Ibuprofen 400 mg", batch="IBU41108", mfg="2024-11-08", exp="2026-10-31",
      mfg_s="08/11/2024", exp_s="31/10/2026", storage_s="Store below 25°C", tmin=None, tmax=25,
      price="M.R.P. Rs.18.40", fx=dict(angle=-22, persp=.12, glare=.4, blur=1.0)),
 dict(t=vial, product="Insulin Glargine Injection 100 IU/mL", batch="IGL7F203", mfg="2025-07-03", exp="2027-06-30",
      mfg_s="03/07/2025", exp_s="30/06/2027", storage_s="Store at 2°C to 8°C. Do not freeze.", tmin=2, tmax=8,
      fx=dict(angle=-6, cyl=.5, shadow=.3, dark=.75, blur=.6)),
 dict(t=carton_flap, product="Amoxicillin 250 mg Capsules", batch="AMX3C551", mfg="2026-02-14", exp="2028-01-31",
      mfg_s="14-FEB-2026", exp_s="31-JAN-2028", storage_s="Store below 25°C in a dry place.", tmin=None, tmax=25,
      fx=dict(angle=90, persp=.06, blur=.5)),
 dict(t=bottle, product="Oral Rehydration Salts Solution", batch="ORS2208K", mfg="2025-08-22", exp="2026-11-30",
      mfg_s="22.08.2025", exp_s="30.11.2026", storage_s="Store below 30°C. Use within 24 h of opening.", tmin=None, tmax=30,
      fx=dict(angle=4, cyl=.3, glare=.5, blur=.8)),
 dict(t=side_panel, product="Pantoprazole 40 mg Tablets", batch="PNT40E19", mfg="2024-05-19", exp="2026-04-30",
      mfg_s="19/05/2024", exp_s="30/04/2026", storage_s="Store below 30°C. Protect from moisture.", tmin=None, tmax=30,
      fx=dict(angle=-9, persp=.18, shadow=.4, dark=.7, blur=.9)),
]
colors = [(200, 40, 50), (20, 110, 80), (0, 80, 160), (120, 40, 140), (220, 120, 0)]
makers = ["Sunrise Pharma Ltd., Baddi (H.P.)", "Medicore Labs Pvt. Ltd., Vapi", "Ganga Biotech Ltd., Hyderabad"]

for i, s in enumerate(S, 1):
    p = dict(s, brand_color=random.choice(colors), maker=random.choice(makers), price=s.get("price", ""))
    lab = to_bgr(s["t"](p))
    fx = s["fx"]
    if "cyl" in fx: lab = cylinder(lab, fx["cyl"])
    img = place(lab, angle=fx.get("angle", 0), persp=fx.get("persp", 0))
    img = light(img, glare=fx.get("glare", 0), shadow=fx.get("shadow", 0), dark=fx.get("dark", 1))
    img = camera(img, blur=fx.get("blur", 0))
    name = f"heldout_{i:02d}"
    cv2.imwrite(f"{OUT}/{name}.jpg", img)
    gt = dict(product=s["product"], batch_no=s["batch"], mfg_date=s["mfg"], expiry_date=s["exp"],
              storage_temp_min=s["tmin"], storage_temp_max=s["tmax"],
              style=s["t"].__name__, effects=fx)
    json.dump(gt, open(f"{OUT}/{name}.json", "w"), indent=2)
print("done")
