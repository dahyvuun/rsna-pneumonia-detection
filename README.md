# rsna-pneumonia-detection

## 📘 Project Overview

This project compares YOLOv8’s built-in Non-Maximum Suppression (NMS) with a custom NMS method using the `ensemble-boxes` library. It is applied to a subset of the RSNA Pneumonia Detection dataset to validate and analyze detection performance.

## 🗂️ Folder Structure

- `nms_compare.py`: Main evaluation script
- `utils.py`: Helper functions for box conversion, matching, metrics
- `results.txt`: Precision/Recall/F1 results (see below)
- `rsna_pneumonia/`: Subset of images/labels + YOLO `.pt` weights

## 🔎 Results

Project: Pneumonia Detection with YOLOv8 + Custom NMS Evaluation
Dataset: RSNA Pneumonia Detection Challenge (100 image subset)
Models tested: YOLOv8n (trained from scratch)
NMS Techniques Compared: Default YOLO-NMS, Soft-NMS, Weighted Boxes Fusion (WBF)
Best Result (WBF):

Precision: 0.86

Recall: 0.92

F1 Score: 0.89


AI Research Project: Pneumonia Detection with YOLOv8 + NMS Techniques

Fine-tuned YOLOv8n on 100-image subset from RSNA dataset

Evaluated NMS variants: Soft-NMS and Weighted Boxes Fusion

Achieved F1 Score of 0.89 using WBF (Precision 0.86, Recall 0.92)

## 🔍 Key Insights from YOLOv8 NMS Comparison on RSNA Pneumonia Subset (100 Images)

1. **WBF significantly outperforms YOLO's built-in NMS**  
   - **F1 Score improved from 0.4483 → 0.8889**
   - This demonstrates that **post-processing methods** like WBF can drastically enhance object detection performance, especially in medical imaging.

2. **YOLO's built-in NMS has high precision but low recall**  
   - YOLO-NMS:
     - Precision: 0.9286  
     - Recall: **0.2955**
   - Indicates that default NMS may **over-suppress detections**, missing many true positives in complex medical images.

3. **WBF offers better balance between precision and recall than Soft-NMS**  
   - F1 Scores:
     - WBF: **0.8889**
     - Soft-NMS: ~0.58
   - WBF merges overlapping boxes instead of discarding them, making it **ideal for subtle or overlapping features** like lung opacities.

4. **Post-processing has major impact — even with the same model**  
   - All evaluations were done using **YOLOv8n** (nano model), yet changing the NMS strategy **nearly doubled detection quality**.
   - Critical takeaway: improving post-processing may be more effective than just switching to larger models.

5. **Efficient and Reproducible Evaluation**  
   - Experiments were done on a **100-image subset** (balanced positives), showing that meaningful insights can be derived **without retraining or large datasets**.
