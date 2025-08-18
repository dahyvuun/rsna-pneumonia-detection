import os
import numpy as np

def convert_yolo_to_ensemble_format(result):
    boxes = []
    scores = []
    labels = []

    h, w = result.orig_shape
    for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
        x1, y1, x2, y2 = box.tolist()
        boxes.append([x1 / w, y1 / h, x2 / w, y2 / h])
        scores.append(conf.item())
        labels.append(int(cls.item()))

    return [boxes], [scores], [labels]

def load_ground_truth(label_path):
    gt_boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                _, x_center, y_center, width, height = map(float, parts)
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                gt_boxes.append([x1, y1, x2, y2])
    return gt_boxes

def match_predictions(pred_boxes, gt_boxes, iou_threshold=0.5):
    matched_gt = set()
    tp = 0
    for pb in pred_boxes:
        best_iou = 0
        best_gt = -1
        for i, gb in enumerate(gt_boxes):
            iou = compute_iou(pb, gb)
            if iou > best_iou:
                best_iou = iou
                best_gt = i
        if best_iou >= iou_threshold and best_gt not in matched_gt:
            tp += 1
            matched_gt.add(best_gt)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def print_metrics(tp, fp, fn, title):
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    print(f"\n== {title} ==")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
