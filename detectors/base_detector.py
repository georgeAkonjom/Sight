from mediapipe.tasks.python import vision

class BaseDetector:
    """Base class for all MediaPipe Task detectors using LIVE_STREAM mode."""
    def __init__(self, model_path, running_mode=vision.RunningMode.LIVE_STREAM):
        self.model_path = model_path
        self.running_mode = running_mode
        self.latest_result = None
        self.detector = None

    def _result_callback(self, result, output_image, timestamp_ms):
        self.latest_result = result

    def process_frame(self, mp_image, timestamp_ms):
        """Process a frame either synchronously (VIDEO mode) or asynchronously (LIVE_STREAM mode)."""
        if not self.detector:
            return
        
        if self.running_mode == vision.RunningMode.LIVE_STREAM:
            self.detector.detect_async(mp_image, timestamp_ms)
        elif self.running_mode == vision.RunningMode.VIDEO:
            self.latest_result = self.detector.detect_for_video(mp_image, timestamp_ms)

    def draw(self, frame):
        """Draw landmarks and annotations on the frame. To be implemented by subclasses."""
        pass

    def get_latest_data(self):
        """Return structured dictionary data from the latest detection. To be implemented by subclasses."""
        return []

    def close(self):
        if self.detector:
            self.detector.close()
