from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.train(
    data="/content/rsna_subset.yaml",
    imgsz=640,
    epochs=10,        
    batch=16,
    workers=2,
    device=0          # GPU
)
