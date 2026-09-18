from ultralytics.models import YOLO
from ultralytics.utils.benchmarks import benchmark

model = YOLO("models/train-small-optimized-adamw-1280/weights/best.pt")

results = model.predict(
    source="dataset/train-custom-scale",  # Path to your folder containing images
    save=True,  # Save the annotated images with bounding boxes
    save_txt=True,  # (Optional) Save results as text files with bounding box coordinates
    conf=0.25,  # (Optional) Confidence threshold
)