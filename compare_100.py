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
import os
import numpy as np
tp_yolo = fp_yolo = fn_yolo = 0
def compute_ious(boxes1, boxes2):
    """
    Compute IoU between each pair of boxes from boxes1 and boxes2.
    boxes1, boxes2: [[x1, y1, x2, y2], ...] in normalized coordinates
    Returns: IoU matrix of shape (len(boxes1), len(boxes2))
    """
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    inter_x1 = np.maximum(boxes1[:, None, 0], boxes2[:, 0])
    inter_y1 = np.maximum(boxes1[:, None, 1], boxes2[:, 1])
    inter_x2 = np.minimum(boxes1[:, None, 2], boxes2[:, 2])
    inter_y2 = np.minimum(boxes1[:, None, 3], boxes2[:, 3])

    inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    union_area = area1[:, None] + area2 - inter_area

    iou = inter_area / (union_area + 1e-8)
    return iou

def load_ground_truth(label_path):
    """
    Load YOLO-format ground truth boxes from a label file.
    Returns a list of normalized [x1, y1, x2, y2] boxes.
    """
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            cls, xc, yc, w, h = map(float, line.strip().split())
            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2
            boxes.append([x1, y1, x2, y2])
    return boxes

for r in results:
    image_name = os.path.basename(r.path).replace(".jpg", ".txt")
    label_path = os.path.join(GROUND_TRUTH_DIR, image_name)

    if not os.path.exists(label_path):
        continue

    gt_boxes = load_ground_truth(label_path)
    pred_boxes = [
        [b[0]/r.orig_shape[1], b[1]/r.orig_shape[0], b[2]/r.orig_shape[1], b[3]/r.orig_shape[0]]
        for b in r.boxes.xyxy.cpu().numpy()
    ]

    tp, fp, fn = match_predictions(pred_boxes, gt_boxes)
    tp_yolo += tp
    fp_yolo += fp
    fn_yolo += fn

precision = tp_yolo / (tp_yolo + fp_yolo + 1e-8)
recall = tp_yolo / (tp_yolo + fn_yolo + 1e-8)
f1 = 2 * precision * recall / (precision + recall + 1e-8)

print("== YOLO Built-in NMS ==")
print(f"TP: {tp_yolo}, FP: {fp_yolo}, FN: {fn_yolo}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
