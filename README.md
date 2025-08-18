# rsna-pneumonia-detection

## 📘 Project Overview

This project compares YOLOv8’s built-in Non-Maximum Suppression (NMS) with a custom NMS method using the `ensemble-boxes` library. It is applied to a subset of the RSNA Pneumonia Detection dataset to validate and analyze detection performance.

## 🗂️ Folder Structure

- `nms_compare.py`: Main evaluation script
- `utils.py`: Helper functions for box conversion, matching, metrics
- `results.txt`: Precision/Recall/F1 results (see below)
- `rsna_pneumonia/`: Subset of images/labels + YOLO `.pt` weights

## 🔎 Results (sample)

Built-in NMS vs. Custom NMS:
