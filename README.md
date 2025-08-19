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
