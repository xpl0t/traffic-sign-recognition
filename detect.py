from os import path
import cv2
import time
from matplotlib.pyplot import box
from ultralytics import YOLO

# Configuration
# source = "/Users/weih/Desktop/DJI_20260215175333_0108_D.MP4"
source = "/home/weih/Videos/circuito/circuito_night.mp4" # 1.mp4"
# source = 0 # Webcam
model_path = "models/train-small-optimized-adamw-1280/weights/best.pt"
output_video_path = "out.mp4"

model = YOLO(model_path)
cap = cv2.VideoCapture(source)
cap_fps = cap.get(cv2.CAP_PROP_FPS)
cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(
    output_video_path,
    fourcc,
    cap_fps,
    (cap_width, cap_height)
)


annotations = "" # YOLO frame annotations
frame_cnt = 0


def update_annotations(results, frame_cnt):
    global annotations

    for box in results[0].boxes:
        cls_id = int(box.cls.item())
        conf_val = float(box.conf.item())
        xywhn = box.xywhn[0].tolist()  # [x_center, y_center, width, height]

        # Format matching YOLO's standard text structure
        line = f"{frame_cnt} {cls_id} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f} {conf_val:.6f}\n"
        annotations += line
    


print("Starting video processing...")
while cap.isOpened():  # Limit to 2000 frames for testing
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False, conf=0.5)

    # Draw bounding boxes onto the CURRENT frame
    if results[0] is not None:
        frame = results[0].plot(img=frame)

        # Save frame annotations to text file
        update_annotations(results, frame_cnt)
        
    video_writer.write(frame)

    print("\rFPS: {:.2f}, Second: {:.2f}, Frame: {}".format(1 / (time.time() - start_time), frame_cnt / cap_fps, frame_cnt), end="")
    frame_cnt += 1


with open(output_video_path.replace(".mp4", ".txt"), "w") as f:
    f.write(annotations)

# Clean up
cap.release()
video_writer.release()
