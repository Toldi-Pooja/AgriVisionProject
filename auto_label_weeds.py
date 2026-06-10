import os
import cv2
import shutil
from pathlib import Path

# =========================================================
# PATHS
# =========================================================
BASE_DIR = Path(r"E:\AgriVision\AgriVisionProject")
SRC_DIR = BASE_DIR / "datasets" / "weed_detect"
DST_DIR = BASE_DIR / "datasets" / "weed_detect_v2"

TRAIN_IMG_SRC = SRC_DIR / "images" / "train"
VAL_IMG_SRC = SRC_DIR / "images" / "val"

TRAIN_IMG_DST = DST_DIR / "images" / "train"
VAL_IMG_DST = DST_DIR / "images" / "val"

TRAIN_LBL_DST = DST_DIR / "labels" / "train"
VAL_LBL_DST = DST_DIR / "labels" / "val"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# =========================================================
# CLASS MAP
# =========================================================
CLASS_MAP = {
    "black_grass": 0,
    "charlock": 1,
    "cleavers": 2,
    "common_chickweed": 3,
    "scentless_mayweed": 4,
}


# =========================================================
# HELPERS
# =========================================================
def ensure_dirs():
    TRAIN_IMG_DST.mkdir(parents=True, exist_ok=True)
    VAL_IMG_DST.mkdir(parents=True, exist_ok=True)
    TRAIN_LBL_DST.mkdir(parents=True, exist_ok=True)
    VAL_LBL_DST.mkdir(parents=True, exist_ok=True)


def reset_destination():
    """
    Old copied images/labels hata do so that clean dataset banega.
    """
    for folder in [TRAIN_IMG_DST, VAL_IMG_DST, TRAIN_LBL_DST, VAL_LBL_DST]:
        folder.mkdir(parents=True, exist_ok=True)
        for item in folder.iterdir():
            if item.is_file():
                item.unlink()


def get_class_id_from_name(stem: str):
    name = stem.lower()

    # handle singular/plural mismatch too
    if name.startswith("black_grass"):
        return 0
    if name.startswith("charlock"):
        return 1
    if name.startswith("cleaver") or name.startswith("cleavers"):
        return 2
    if name.startswith("common_chickweed"):
        return 3
    if name.startswith("scentless_mayweed"):
        return 4

    return None


def find_green_bbox(img):
    """
    Detect green plant area and return bbox in YOLO format.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # slightly broader green range
    lower_green = (20, 20, 20)
    upper_green = (100, 255, 255)
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    img_h, img_w = img.shape[:2]
    img_area = img_h * img_w

    good = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > img_area * 0.0015:   # tiny noise ignore
            good.append(c)

    if not good:
        return None

    x_min, y_min = img_w, img_h
    x_max, y_max = 0, 0

    for c in good:
        x, y, w, h = cv2.boundingRect(c)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    w = x_max - x_min
    h = y_max - y_min

    if w <= 0 or h <= 0:
        return None

    # padding
    pad_x = int(0.04 * img_w)
    pad_y = int(0.04 * img_h)

    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(img_w, x_max + pad_x)
    y_max = min(img_h, y_max + pad_y)

    w = x_max - x_min
    h = y_max - y_min

    # reject full-image boxes
    if w / img_w > 0.98 and h / img_h > 0.98:
        return None

    x_center = (x_min + w / 2) / img_w
    y_center = (y_min + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h

    return x_center, y_center, w_norm, h_norm


def process_split(img_src_dir, img_dst_dir, lbl_dst_dir, split_name):
    total = 0
    labeled = 0
    skipped = 0

    for file in img_src_dir.iterdir():
        if file.suffix.lower() not in IMAGE_EXTS:
            continue

        total += 1
        class_id = get_class_id_from_name(file.stem)

        if class_id is None:
            print(f"[SKIP CLASS] {file.name}")
            skipped += 1
            continue

        img = cv2.imread(str(file))
        if img is None:
            print(f"[SKIP READ] {file.name}")
            skipped += 1
            continue

        bbox = find_green_bbox(img)
        if bbox is None:
            print(f"[NO BBOX] {file.name}")
            skipped += 1
            continue

        x_center, y_center, w_norm, h_norm = bbox

        shutil.copy2(str(file), str(img_dst_dir / file.name))

        txt_path = lbl_dst_dir / f"{file.stem}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        labeled += 1
        print(f"[OK] {split_name}: {file.name}")

    print(f"\n{split_name} summary -> total: {total}, labeled: {labeled}, skipped: {skipped}\n")


def create_yaml():
    yaml_path = DST_DIR / "data.yaml"
    content = f"""path: {DST_DIR.as_posix()}
train: images/train
val: images/val

names:
  0: black_grass
  1: charlock
  2: cleavers
  3: common_chickweed
  4: scentless_mayweed
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[OK] data.yaml created at: {yaml_path}")


if __name__ == "__main__":
    ensure_dirs()
    reset_destination()
    process_split(TRAIN_IMG_SRC, TRAIN_IMG_DST, TRAIN_LBL_DST, "train")
    process_split(VAL_IMG_SRC, VAL_IMG_DST, VAL_LBL_DST, "val")
    create_yaml()
    print("Done.")