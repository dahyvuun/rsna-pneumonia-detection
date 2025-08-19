import os
import numpy as np
from ensemble_boxes import weighted_boxes_fusion

# === Load ground truth from YOLO format labels ===
def load_ground_truth(label_path):
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                _, x_center, y_center, w, h = map(float, parts)
                x1 = x_center - w / 2
                y1 = y_center - h / 2
                x2 = x_center + w / 2
                y2 = y_center + h / 2
                boxes.append([x1, y1, x2, y2])
    return boxes

# === IoU computation ===
def compute_ious(box, boxes):
    x1 = np.maximum(box[:, None, 0], boxes[:, 0])
    y1 = np.maximum(box[:, None, 1], boxes[:, 1])
    x2 = np.minimum(box[:, None, 2], boxes[:, 2])
    y2 = np.minimum(box[:, None, 3], boxes[:, 3])
    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area[:, None] + boxes_area - inter_area
    return inter_area / np.maximum(union_area, 1e-8)

# === Matching prediction vs ground truth ===
def match_predictions(pred_boxes, gt_boxes, iou_threshold=0.5):
    matched_gt = set()
    tp = 0
    for pb in pred_boxes:
        matched = False
        for i, gb in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            if compute_ious(np.array([pb]), np.array([gb]))[0][0] >= iou_threshold:
                matched_gt.add(i)
                matched = True
                break
        if matched:
            tp += 1
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn

# === WBF Evaluation ===
GROUND_TRUTH_DIR = "/content/drive/MyDrive/rsna_pneumonia/yolo_subset_100/labels"  # adjust if needed
tp_wbf = fp_wbf = fn_wbf = 0

for r in results:
    image_path = r.path
    image_name = os.path.basename(image_path).replace(".jpg", ".txt")
    label_path = os.path.join(GROUND_TRUTH_DIR, image_name)
    
    if not os.path.exists(label_path):
        continue
    
    gt_boxes = load_ground_truth(label_path)
    image_width, image_height = r.orig_shape[1], r.orig_shape[0]

    # 🧠 Normalize predicted boxes
    boxes = [[
        [b[0] / image_width, b[1] / image_height, b[2] / image_width, b[3] / image_height]
        for b in r.boxes.xyxy.cpu().numpy()
    ]]
    scores = [[conf.item() for conf in r.boxes.conf]]
    labels = [[int(cls.item()) for cls in r.boxes.cls]]

    if not boxes[0]:
        continue  # skip if no predictions

    # 🧠 Apply WBF
    fused_boxes, fused_scores, _ = weighted_boxes_fusion(boxes, scores, labels, iou_thr=0.5, skip_box_thr=0.001)

    # ✅ Match against GT
    tp, fp, fn = match_predictions(fused_boxes, gt_boxes, iou_threshold=0.5)
    tp_wbf += tp
    fp_wbf += fp
    fn_wbf += fn

# === Final Metrics ===
precision = tp_wbf / (tp_wbf + fp_wbf + 1e-8)
recall = tp_wbf / (tp_wbf + fn_wbf + 1e-8)
f1 = 2 * precision * recall / (precision + recall + 1e-8)

print("== Weighted Boxes Fusion (WBF) Results ==")
print(f"TP: {tp_wbf}, FP: {fp_wbf}, FN: {fn_wbf}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
