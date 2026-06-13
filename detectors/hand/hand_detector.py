import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from detectors.base_detector import BaseDetector
from helpers.coordinate_translator import CoordinateTranslator

class HandDetectorWrapper(BaseDetector):
    def __init__(self, model_path='models/hand_landmarker.task', running_mode=vision.RunningMode.LIVE_STREAM):
        super().__init__(model_path, running_mode)
        
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=self.model_path),
            running_mode=self.running_mode,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            result_callback=self._result_callback if self.running_mode == vision.RunningMode.LIVE_STREAM else None
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        self.translator = CoordinateTranslator()
        
        # Rendering utilities
        self.mp_drawing = mp.tasks.vision.drawing_utils
        self.mp_drawing_styles = mp.tasks.vision.drawing_styles
        self.mp_hands = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

    def get_latest_data(self):
        if not self.latest_result or not self.latest_result.hand_landmarks:
            return []

        hands_data = []
        for idx, hand_landmarks in enumerate(self.latest_result.hand_landmarks):
            # Guard against mismatched lists or missing handedness classifications
            if idx >= len(self.latest_result.handedness) or not self.latest_result.handedness[idx]:
                continue
                
            raw_handedness = self.latest_result.handedness[idx][0].category_name
            handedness = "Right" if raw_handedness == "Left" else "Left"

            # Get geometry
            hand_data = self.translator.get_structured_data(hand_landmarks, handedness)
            
            hands_data.append({
                'handedness': handedness,
                'landmarks': hand_data['landmarks'],
                'raw_landmarks': hand_landmarks  # Preserved for local drawing utilities
            })
        return hands_data

    def draw(self, frame):
        hands_data = self.get_latest_data()
        if not hands_data:
            return

        for idx, hand in enumerate(hands_data):
            # Display Results
            text_y = 50 + (idx * 80)
            wrist = hand['landmarks'][0]
            
            cv2.putText(frame, f"{hand['handedness']}: x={wrist['x']:.2f}, y={wrist['y']:.2f}", 
                        (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if self.mp_drawing:
                self.mp_drawing.draw_landmarks(
                    frame, hand['raw_landmarks'], self.mp_hands,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
