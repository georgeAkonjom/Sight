import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from detectors.base_detector import BaseDetector
from helpers.coordinate_translator import CoordinateTranslator

class PoseDetectorWrapper(BaseDetector):
    def __init__(self, model_path='models/pose_landmarker.task', running_mode=vision.RunningMode.LIVE_STREAM):
        self.translator = CoordinateTranslator()
        
        try:
            super().__init__(model_path, running_mode)
            options = vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=self.model_path),
                running_mode=self.running_mode,
                result_callback=self._result_callback if self.running_mode == vision.RunningMode.LIVE_STREAM else None
            )
            self.detector = vision.PoseLandmarker.create_from_options(options)
        except Exception as e:
            print(f"Pose Detector initialization failed (likely missing model): {e}")
            self.detector = None
        
        # Rendering utilities
        self.mp_drawing = mp.tasks.vision.drawing_utils
        self.mp_drawing_styles = mp.tasks.vision.drawing_styles
        self.mp_pose = mp.tasks.vision.PoseLandmarksConnections

    def get_latest_data(self):
        if not self.latest_result or not self.latest_result.pose_landmarks:
            return []

        poses_data = []
        for idx, pose_landmarks in enumerate(self.latest_result.pose_landmarks):
            # Parse normal landmarks
            landmarks = self.translator.get_landmarks(pose_landmarks)
            
            # Parse world landmarks if available
            world_landmarks = []
            if self.latest_result.pose_world_landmarks and idx < len(self.latest_result.pose_world_landmarks):
                world_landmarks = self.translator.get_landmarks(self.latest_result.pose_world_landmarks[idx])

            poses_data.append({
                'landmarks': landmarks,
                'world_landmarks': world_landmarks,
                'raw_landmarks': pose_landmarks
            })
        return poses_data

    def draw(self, frame):
        poses_data = self.get_latest_data()
        if not poses_data:
            return

        for pose in poses_data:
            self.mp_drawing.draw_landmarks(
                frame,
                pose['raw_landmarks'],
                self.mp_pose.POSE_LANDMARKS,
                self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
