# compare.py

from ultralytics import YOLO
from ensemble_boxes import nms
import os
from utils import convert_yolo_to_ensemble_format, load_ground_truth, match_predictions, print_metrics
from PIL import Image

# === CONFIG ===
MODEL_PATH = "best.pt"
IMAGE_DIR = "/content/drive/MyDrive/rsna_pneumonia/yolo_subset/images/val"
GROUND_TRUTH_DIR = "/content/drive/MyDrive/rsna_pneumonia/yolo_subset/labels"
CONF_THRESHOLD = 0.1
IOU_THRESHOLD = 0.5

# === LOAD MODEL ===
model = YOLO(MODEL_PATH)
results = model.predict(source=IMAGE_DIR, conf=CONF_THRESHOLD, save=False)

# === COUNTING PREDICTIONS ===
total_yolo = 0
total_custom = 0
image_count = 0

tp_yolo = fp_yolo = fn_yolo = 0
tp_custom = fp_custom = fn_custom = 0

for i, r in enumerate(results):
    boxes_list, scores_list, labels_list = convert_yolo_to_ensemble_format(r)

    if len(boxes_list[0]) == 0:
        print(f"Skipping Image {i} — No predictions")
        continue

    # Custom NMS
    boxes_nms, scores_nms, labels_nms = nms(boxes_list, scores_list, labels_list, iou_thr=IOU_THRESHOLD)
    total_yolo += len(r.boxes)
    total_custom += len(boxes_nms)
    image_count += 1

print(f"\nProcessed {image_count} images with predictions.")
print(f"Total YOLO Boxes: {total_yolo}")
print(f"Total Custom NMS Boxes: {total_custom}")

# === METRICS EVALUATION ===
for i, r in enumerate(results):
    image_path = r.path
    image_name = os.path.basename(image_path).replace(".png", ".txt")
    label_path = os.path.join(GROUND_TRUTH_DIR, image_name)

    if not os.path.exists(label_path):
        continue

    gt_boxes = load_ground_truth(label_path)

    # Built-in YOLO
    yolo_boxes = r.boxes.xyxy.cpu().numpy()
    yolo_boxes_norm = [[b[0]/r.orig_shape[1], b[1]/r.orig_shape[0], b[2]/r.orig_shape[1], b[3]/r.orig_shape[0]] for b in yolo_boxes]
    tp, fp, fn = match_predictions(yolo_boxes_norm, gt_boxes, iou_threshold=IOU_THRESHOLD)
    tp_yolo += tp
    fp_yolo += fp
    fn_yolo += fn

    # Custom NMS
    boxes_list, scores_list, labels_list = convert_yolo_to_ensemble_format(r)
    if len(boxes_list[0]) == 0:
        continue

    boxes_nms, _, _ = nms(boxes_list, scores_list, labels_list, iou_thr=IOU_THRESHOLD)
    tp, fp, fn = match_predictions(boxes_nms, gt_boxes, iou_threshold=IOU_THRESHOLD)
    tp_custom += tp
    fp_custom += fp
    fn_custom += fn

# === PRINT FINAL METRICS ===
print_metrics(tp_yolo, fp_yolo, fn_yolo, "YOLO Built-in NMS")
print_metrics(tp_custom, fp_custom, fn_custom, "Custom NMS (ensemble-boxes)")