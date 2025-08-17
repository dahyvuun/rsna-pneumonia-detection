# RSNA Pneumonia Detection using YOLOv8

This project detects pneumonia in chest X-ray images using the RSNA dataset and YOLOv8.

## 📁 Structure

- `preprocess.py`: Converts RSNA bounding boxes into YOLOv8-compatible format
- `train_yolo.py`: Trains YOLOv8 model on the processed dataset
- `rsna.yaml`: YOLO dataset config file
- `requirements.txt`: Python dependencies

## 📦 Dataset

- Source: [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/)
- Preprocessing creates a balanced subset of positive/negative samples

## 🛠️ How to Run

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt

2. Run preprocessing
    
    python preprocess.py

3. Train the YOLOv8 model:

    python train_yolo.py