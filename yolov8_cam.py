from ultralytics import YOLO
model = YOLO("yolov8n.pt")

results = model(R"C:\Users\kriip\Documents\innovation\yolo\images\test.mp4", save=True) 