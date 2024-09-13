#Ultralytics YOLOv8.2.66 🚀 Python-3.10.14 torch-2.3.1

from ultralytics import YOLO

model = YOLO("yolov8x-seg.pt")  # load a pretrained YOLOv8n segmentation model

model.train(
    data="coco128-seg.yaml",
    imgsz=704,
    device="0",
    epochs=200,
    batch=4,
    patience=15,
    amp=True,
    project="/home/ubuntu/training/cella/training/experiment",
    name="square-sk-yolov9",
    lr0=0.01,
    optimizer="SGD",
    seed=7,
    mosaic=0
)
