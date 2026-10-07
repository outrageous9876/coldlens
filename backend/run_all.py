import sys
from pathlib import Path
import cv2
from preprocess import preprocess, save_steps

photos = ["test1.webp", "test2.jpg", "test3.jpeg"]
for name in photos:
    path = Path("../data/raw") / name
    image = cv2.imread(str(path))
    if image is None:
        print(f"{name}: FAILED TO LOAD")
        continue
    _, steps = preprocess(image)
    out_dir = Path("../data/debug") / Path(name).stem
    saved = save_steps(steps, out_dir)
    print(f"{name}: saved to {out_dir}")
