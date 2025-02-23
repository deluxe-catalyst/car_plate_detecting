from ultralytics import YOLO
import cv2


model = YOLO("best.pt")

# Путь к обрабатываемому изоображению
image_path = "input.jpeg"
results = model(image_path)  # Запускаем предсказание

for result in results:
    result.show()

results[0].save(filename="ProjectLabel/scale_1200_annotated.jpeg")