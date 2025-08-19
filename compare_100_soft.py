import os
import numpy as np

# === Ground Truth Loader ===
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

# === IoU Computation ===
def compute_ious(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - inter_area
    return inter_area / np.maximum(union_area, 1e-8)

# === Soft-NMS ===
def soft_nms(boxes, scores, sigma=0.5, conf_thresh=0.05):
    boxes = np.array(boxes)
    scores = np.array(scores)
    keep = []

    while len(scores) > 0:
        max_idx = np.argmax(scores)
        max_box = boxes[max_idx]
        max_score = scores[max_idx]
        keep.append((max_box.tolist(), float(max_score)))

        boxes = np.delete(boxes, max_idx, axis=0)
        scores = np.delete(scores, max_idx)

        if len(boxes) == 0:
            break

        ious = compute_ious(max_box, boxes)
        scores = scores * np.exp(-(ious ** 2) / sigma)

        keep_mask = scores > conf_thresh
        boxes = boxes[keep_mask]
        scores = scores[keep_mask]

    filtered_boxes, _ = zip(*keep) if keep else ([], [])
    return list(filtered_boxes)

# === Matching prediction vs ground truth ===
def match_predictions(pred_boxes, gt_boxes, iou_threshold=0.5):
    matched_gt = set()
    tp = 0
    for pb in pred_boxes:
        matched = False
        for i, gb in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            if compute_ious(pb, np.array([gb]))[0] >= iou_threshold:
                matched_gt.add(i)
                matched = True
                break
        if matched:
            tp += 1
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn

# === Soft-NMS Evaluation ===
GROUND_TRUTH_DIR = "/content/drive/MyDrive/rsna_pneumonia/yolo_subset_100/labels"
tp_softnms = fp_softnms = fn_softnms = 0

for r in results:
    image_path = r.path
    image_name = os.path.basename(image_path).replace(".jpg", ".txt")
    label_path = os.path.join(GROUND_TRUTH_DIR, image_name)

    if not os.path.exists(label_path):
        continue

    gt_boxes = load_ground_truth(label_path)
    image_width, image_height = r.orig_shape[1], r.orig_shape[0]

    # Normalize YOLO outputs
    boxes = [
        [b[0] / image_width, b[1] / image_height, b[2] / image_width, b[3] / image_height]
        for b in r.boxes.xyxy.cpu().numpy()
    ]
    scores = [conf.item() for conf in r.boxes.conf]

    if not boxes:
        continue

    # Apply Soft-NMS
    filtered_boxes = soft_nms(boxes, scores, sigma=0.5, conf_thresh=0.05)

    # Match with GT
    tp, fp, fn = match_predictions(filtered_boxes, gt_boxes, iou_threshold=0.5)
    tp_softnms += tp
    fp_softnms += fp
    fn_softnms += fn

# === Final Metrics ===
precision = tp_softnms / (tp_softnms + fp_softnms + 1e-8)
recall = tp_softnms / (tp_softnms + fn_softnms + 1e-8)
f1 = 2 * precision * recall / (precision + recall + 1e-8)

print("== Soft-NMS Results ==")
print(f"TP: {tp_softnms}, FP: {fp_softnms}, FN: {fn_softnms}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
