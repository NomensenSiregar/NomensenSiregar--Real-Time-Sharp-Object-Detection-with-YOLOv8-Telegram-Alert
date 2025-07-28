from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # base model
model.train(
    data="dataset_ml/data.yaml",
    epochs=10,
    imgsz=640,
    name="knife_scissors_custom"
)
