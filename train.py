from ultralytics import YOLO
import torch

# for MacOS
# https://pytorch.org/docs/stable/notes/mps.html

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = YOLO("yolov8n.pt")
model.to(device)
model.train(data="data.yaml", epochs=100, batch=32, imgsz=640)