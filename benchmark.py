from ultralytics.utils.benchmarks import benchmark

import config

# models = [
#     "models/train-nano/weights/best.pt",
#     "models/train-small/weights/best.pt",
#     "models/train-medium/weights/best.pt",
#     "models/train-large/weights/best.pt",
# ]
# models = [
#     "models/train-nano/weights/best_float32.pt",
#     "models/train-small/weights/best_float32.pt",
#     "models/train-medium/weights/best_float32.pt",
#     "models/train-large/weights/best_float32.pt",
# ]
models = [
    "models/train-small-optimized-adamw/weights/best.pt",
    "models/train-small/weights/best.pt",
    "models/train-small-optimized/weights/best.pt",
    ]

for model in models:
    # Use torchscript format for amd rocm backend and coreml format for apple mps backend
    benchmark(model=model, task="detect", data="dataset.yml", imgsz=640, device=config.DEVICE, format=config.BENCH_FORMAT)
