import cv2
import threading
import time
from ultralytics import YOLO

# Configuration
# source = "/Users/weih/Desktop/DJI_20260215175333_0108_D.MP4"
source = "/home/weih/Videos/1.mp4"
# source = 0 # Uncomment for webcam
model_path = "models/train-medium/weights/best.pt"
output_video_path = "out.mp4"

class AsyncYOLODetector:
    def __init__(self, source=0, model_name='yolov8n.pt', output_path='output.mp4'):
        self.source = source
        self.output_path = output_path
        
        # Load the YOLO model
        self.model = YOLO(model_name)
        self.cap = cv2.VideoCapture(source)
        
        # Shared variables between threads
        self.latest_frame = None
        self.latest_results = None
        self.yolo_fps = 0.0
        self.video_fps = 0.0
        
        # Thread locks for thread-safe reading/writing
        self.frame_lock = threading.Lock()
        self.results_lock = threading.Lock()
        
        self.running = True
        self.paused = False
        self.video_writer = None
        
        # Start the background YOLO inference thread
        self.inference_thread = threading.Thread(target=self._yolo_worker, daemon=True)
        self.inference_thread.start()

    def _yolo_worker(self):
        last_processed_frame = None
        
        while self.running:
            # Safely grab the latest frame
            with self.frame_lock:
                if self.latest_frame is None or self.latest_frame is last_processed_frame:
                    frame_to_process = None
                else:
                    frame_to_process = self.latest_frame.copy()
                    last_processed_frame = self.latest_frame

            # If no new frame is available or paused, yield CPU time briefly
            if frame_to_process is None or self.paused:
                time.sleep(0.005)
                continue

            start_time = time.time()
            
            # Run YOLO inference
            results = self.model(frame_to_process, verbose=False, conf=0.5)
            
            # Safely update the latest results
            with self.results_lock:
                self.latest_results = results[0]
                
            # Calculate YOLO processing FPS
            process_time = time.time() - start_time
            if process_time > 0:
                self.yolo_fps = 1.0 / process_time

    def _mouse_callback(self, event, x, y, flags, param):
        # Pause/Unpause on left mouse click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.paused = not self.paused

    def run(self):
        # Setup OpenCV Window and Mouse Callback
        window_name = "Async YOLO Real-Time"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._mouse_callback)

        source_fps = self.cap.get(cv2.CAP_PROP_FPS)
        delay = int(1000 / source_fps) if source_fps > 0 and isinstance(self.source, str) else 1
        
        prev_time = time.time()

        while self.cap.isOpened() and self.running:
            if self.paused:
                # If paused, wait for key press to allow quitting, but don't read new frames
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    self.running = False
                    break
                # Reset prev_time so FPS calculation doesn't plummet when unpausing
                prev_time = time.time()
                continue

            ret, frame = self.cap.read()
            if not ret:
                break

            # Scale to 640 width (maintain aspect ratio)
            # target_width = 640
            # h, w = frame.shape[:2]
            # target_height = int(h * (target_width / w))
            # frame = cv2.resize(frame, (target_width, target_height))

            # Initialize VideoWriter once we know the resized frame dimensions
            if self.video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps_out = source_fps if source_fps > 0 else 30.0
                self.video_writer = cv2.VideoWriter(
                    self.output_path, fourcc, fps_out, (1920, 1080) #, (target_width, target_height)
                )

            # Update the latest frame for the background thread to process
            with self.frame_lock:
                self.latest_frame = frame

            # Prepare the display frame
            display_frame = frame.copy()

            # Safely grab the latest bounding box results
            with self.results_lock:
                current_results = self.latest_results

            # Draw bounding boxes onto the CURRENT frame
            if current_results is not None:
                display_frame = current_results.plot(img=display_frame)

            # Calculate Video loop FPS (exponential moving average for smoothing)
            curr_time = time.time()
            loop_time = curr_time - prev_time
            if loop_time > 0:
                current_video_fps = 1.0 / loop_time
                # Smooth the FPS value
                self.video_fps = 0.9 * self.video_fps + 0.1 * current_video_fps
            prev_time = curr_time

            # Draw FPS metrics
            cv2.putText(display_frame, f"YOLO FPS: {self.yolo_fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Video FPS: {self.video_fps:.1f}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            
            if self.paused:
                 cv2.putText(display_frame, "PAUSED", (target_width // 2 - 50, target_height // 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # Write to video file
            self.video_writer.write(display_frame)

            cv2.imshow(window_name, display_frame)

            # Quit if 'q' is pressed
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                self.running = False
                break

        # Clean up
        self.cap.release()
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = AsyncYOLODetector(source=source, model_name=model_path, output_path=output_video_path)
    detector.run()