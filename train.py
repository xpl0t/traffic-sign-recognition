from ultralytics import YOLO
import config

# Load a model
model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="dataset.yml", epochs=100, imgsz=640, device=config.DEVICE)