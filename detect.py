import cv2
import threading
import time
from ultralytics import YOLO

# Configuration
# source = "/Users/weih/Desktop/DJI_20260215175333_0108_D.MP4"
source = "/home/weih/Videos/1.mp4"
# source = 0 # Uncomment for webcam
model_path = "models/train-small-optimized-adamw/weights/best.pt"
output_video_path = "out.mp4"

model = YOLO(model_path)
cap = cv2.VideoCapture(source)
# cap.set(cv2.CAP_PROP_FPS, 60)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(
    output_video_path,
    fourcc,
    cap.get(cv2.CAP_PROP_FPS),
    #(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    (1920, 1080)
)

print("Starting video processing...")

cnt = 0
while cap.isOpened() and cnt < 3000:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break
    cnt += 1

    results = model(frame, verbose=False, conf=0.5)

    # Draw bounding boxes onto the CURRENT frame
    if results[0] is not None:
        frame = results[0].plot(img=frame)

    video_writer.write(frame)

    print("\rFPS: {:.2f}".format(1 / (time.time() - start_time)), end="")


# Clean up
cap.release()
video_writer.release()
