from ultralytics import YOLO

import config

model = YOLO("yolo26s.pt")

results = model.tune(
    data="dataset.yml",
    epochs=10,
    iterations=30,
    optimizer="AdamW",
    plots=True,
    save=True,
    device=config.DEVICE
)