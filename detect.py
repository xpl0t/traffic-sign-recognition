import cv2
import threading
import time
from ultralytics import YOLO

# source = 0
source = "/Users/weih/Desktop/DJI_20260215175333_0108_D.MP4"
model_path = "models/train-small/weights/best_float32.pt"

class AsyncYOLODetector:
    def __init__(self, source=0, model_name='yolov8n.pt'):
        self.source = source
        # Load the YOLOv8 model
        self.model = YOLO(model_name)
        
        self.cap = cv2.VideoCapture(source)
        
        # Shared variables between threads
        self.latest_frame = None
        self.latest_results = None
        self.yolo_fps = 0.0
        
        # Thread locks for thread-safe reading/writing
        self.frame_lock = threading.Lock()
        self.results_lock = threading.Lock()
        
        self.running = True
        
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

            # If no new frame is available, yield CPU time briefly
            if frame_to_process is None:
                time.sleep(0.005)
                continue

            start_time = time.time()
            
            # Run YOLO inference
            # verbose=False prevents the console from being flooded with print statements
            results = self.model(frame_to_process, verbose=False)
            
            # Safely update the latest results
            with self.results_lock:
                self.latest_results = results[0]
                
            # Calculate YOLO processing FPS
            process_time = time.time() - start_time
            if process_time > 0:
                self.yolo_fps = 1.0 / process_time

    def run(self):
        # Calculate playback delay to match video FPS (if using a file)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        delay = int(1000 / fps) if fps > 0 and isinstance(self.source, str) else 1

        while self.cap.isOpened() and self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Update the latest frame for the background thread to process
            with self.frame_lock:
                self.latest_frame = frame

            # Prepare the display frame
            display_frame = frame.copy()

            # Safely grab the latest bounding box results
            with self.results_lock:
                current_results = self.latest_results

            # If we have results, draw them onto the CURRENT frame
            # This causes boxes to temporarily trail behind fast objects if YOLO is slow,
            # but guarantees the video itself never stutters.
            if current_results is not None:
                display_frame = current_results.plot(img=display_frame)

            # Draw YOLO processing FPS
            fps_text = f"YOLO FPS: {self.yolo_fps:.1f}"
            cv2.putText(display_frame, fps_text, (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            cv2.imshow("Async YOLO Real-Time", display_frame)

            # Quit if 'q' is pressed
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                self.running = False
                break

        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # For Live Webcam: source = 0
    # For Video File: source = "path/to/your/video.mp4"
    detector = AsyncYOLODetector(source=source, model_name=model_path)
    detector.run()