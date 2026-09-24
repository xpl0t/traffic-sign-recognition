from queue import Queue
import cv2
import threading
import time
from ultralytics import YOLO

# Configuration
# source = "/Users/weih/Desktop/DJI_20260215175333_0108_D.MP4"
source = "/home/weih/Videos/circuito/circuito_sunset.mp4" # 1.mp4"
# source = 0 # Webcam
model_path = "models/train-small-optimized-adamw-1280/weights/best.pt"

# YOLO model and opencv video capture
model = YOLO(model_path)
cap = cv2.VideoCapture(source)
cap_fps = cap.get(cv2.CAP_PROP_FPS)
cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Shared variables
stream_finished = False
frame = None
frame_lock = threading.Lock()
results = None
results_lock = threading.Lock()
frame_cnt = 0 # Frame cnt does not require a lock, as it is only updated in the video_stream() thread.

# Optional video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(
    "out.mp4",
    fourcc,
    cap_fps,
    (cap_width, cap_height)
)


def yolo_detection_loop():
    global model, stream_finished, frame, frame_lock, frame_cnt, results, results_lock

    while not stream_finished:
        frame_copy = None

        # Wait until next frame available
        while not stream_finished:
            with frame_lock:
                if frame is not None:
                    frame_copy = frame.copy()
                    frame = None
                    break
            time.sleep(0.01)

        if stream_finished:
            break


        # Detect objects in the frame
        start_time = time.time()
        res = model(frame_copy, verbose=False, conf=0.5)
        print("\rYOLO FPS: {:.2f}, Second: {:.2f}, Frame: {}".format(1 / (time.time() - start_time), frame_cnt / cap_fps, frame_cnt), end="")


        # Set results in global results variable
        with results_lock:
            results = res

    print() # New line for fps counter

def video_stream():
    global cap, stream_finished, frame, frame_lock, frame_cnt, results, results_lock

    while cap.isOpened():

        start_time = time.time()

        ret, cur_frame = cap.read()
        if not ret:
            break

        with frame_lock:
            frame = cur_frame

        with results_lock:
            if results is not None:
                # Draw bounding boxes onto the current frame
                cur_frame = results[0].plot(img=cur_frame)

        cv2.imshow("live", cur_frame)
        video_writer.write(cur_frame)

        end_time = time.time()

        # Quit if 'q' is pressed
        if cv2.waitKey(max(1, int((1 / 30) * 1000 - (end_time - start_time) * 1000))) & 0xFF == ord('q'):
            break

        frame_cnt += 1

    stream_finished = True


# Start the YOLO detection thread
detect_th = threading.Thread(target=yolo_detection_loop)
detect_th.start()

# Start video streaming
video_stream()

# Cleanup
cap.release()
cv2.destroyAllWindows()
video_writer.release()

# Join YOLO detection thread and wait for exit
detect_th.join()