from ultralytics.utils.benchmarks import benchmark

models = [
    "runs/detect/train-nano/weights/best.pt",
    "runs/detect/train-small/weights/best.pt",
    "runs/detect/train-medium/weights/best.pt",
    "runs/detect/train-large/weights/best.pt",
    "runs/detect/train-xxl-incomplete/weights/best.pt",
]

for model in models:
    # Use only torchscript format, as most of the other formats require nvidia gpu
    benchmark(model=model, data="dataset.yml", imgsz=640, device=0, format="torchscript")
