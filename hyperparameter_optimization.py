from ultralytics import YOLO

model = YOLO("yolo26n.pt")

results = model.tune(
    data="dataset.yml",
    epochs=10,
    iterations=30,
    optimizer="AdamW",
    plots=True,
    save=True,
    device=0
)