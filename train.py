from ultralytics import YOLO
import config

# Load a model
model = YOLO("yolo26s.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(
    data="dataset.yml",
    epochs=100,
    imgsz=640,
    device=config.DEVICE,
    optimizer="AdamW",
    lr0=0.00151,
    lrf=0.0158,
    momentum=0.85434,
    weight_decay=0.00066,
    warmup_epochs=2.95496,
    warmup_momentum=0.95,
    box=10.39805,
    cls=0.57884,
    cls_pw=0.0008,
    dfl=1.11783,
    hsv_h=0.0077,
    hsv_s=0.71044,
    hsv_v=0.41176,
    degrees=0.00078,
    translate=0.08651,
    scale=0.26183,
    shear=0.00162,
    perspective=0.00074,
    flipud=0.00183,
    fliplr=0.57479,
    bgr=0.01247,
    mosaic=0.95407,
    mixup=0.03001,
    cutmix=0.0145,
    copy_paste=0.00255,
    close_mosaic=9)
