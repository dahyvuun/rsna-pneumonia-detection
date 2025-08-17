import pandas as pd, numpy as np, random, glob, os, shutil, cv2
from pathlib import Path

CSV_PATH = "/content/drive/MyDrive/rsna_pneumonia/data/stage2_train_metadata.csv"
IMG_DIR = "/content/drive/MyDrive/rsna_pneumonia/data/Training/Images"
YOLO_DIR = "/content/rsna_yolo_subset"

# Load the CSV and match it to available PNGs
df = pd.read_csv(CSV_PATH)
df["patientId"] = df["patientId"].astype(str)
png_ids = {Path(p).stem: p for p in glob.glob(f"{IMG_DIR}/*.png")}
df = df[df.patientId.isin(png_ids.keys())].copy()

# Pick balanced mini set (adjust numbers if needed)
pos_ids = df[df.Target==1].patientId.unique().tolist()
neg_ids = list(set(df.patientId.unique()) - set(pos_ids))
random.seed(42)
N_POS, N_NEG = 120, 120
ids = np.array(
    random.sample(pos_ids, min(N_POS, len(pos_ids))) +
    random.sample(neg_ids, min(N_NEG, len(neg_ids)))
)

# Split train/val
rng = np.random.default_rng(42)
rng.shuffle(ids)
split = int(0.9 * len(ids))
train_ids, val_ids = set(ids[:split]), set(ids[split:])

# Group data by patientId
g = df[df.patientId.isin(ids)].groupby("patientId")

# Create YOLO folders
for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
    Path(f"{YOLO_DIR}/{sub}").mkdir(parents=True, exist_ok=True)

# Label writer
def write_yolo_label(txt_path, boxes, w, h):
    with open(txt_path, "w") as f:
        for (x,y,bw,bh) in boxes:
            cx = (x + bw/2) / w
            cy = (y + bh/2) / h
            nw = bw / w
            nh = bh / h
            f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

for pid in ids:
    src = png_ids[pid]
    img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Unreadable:", src)
        continue
    h, w = img.shape[:2]

    boxes = []
    if pid in g.groups:
        for _, r in g.get_group(pid).iterrows():
            if int(r.Target) == 1:
                boxes.append((float(r.x), float(r.y), float(r.width), float(r.height)))

    if not boxes:
        continue  # optional: skip images without labels

    split_dir = "train" if pid in train_ids else "val"

    dst_img = f"{YOLO_DIR}/images/{split_dir}/{pid}.png"
    shutil.copy2(src, dst_img)

    dst_lab = f"{YOLO_DIR}/labels/{split_dir}/{pid}.txt"
    write_yolo_label(dst_lab, boxes, w, h)

