from gesture_recognition.hand_gestures import recognize_hand_gestures
from gesture_recognition.pose_gestures import recognize_pose_gestures
from gesture_recognition.body_language import BodyLanguageRecognizer

class GestureEvaluator:
    """
    High-level orchestrator that consumes structured data from Hand, Pose, and Face detectors
    and returns a unified dictionary of active gestures and body language cues.
    """
    def __init__(self, calibration_duration_ms=10000):
        self.body_language_recognizer = BodyLanguageRecognizer(calibration_duration_ms)

    def evaluate(self, hands_data=None, pose_data=None, face_data=None, timestamp_ms=None):
        """
        Evaluate current tracking frame data.
        
        hands_data: list of hand landmarks and handedness (from HandDetectorWrapper)
        pose_data: list of pose landmarks (from PoseDetectorWrapper)
        face_data: list of face blendshapes and landmarks (from FaceDetectorWrapper)
        timestamp_ms: frame timestamp in milliseconds
        
        Returns:
            dict containing:
                - 'hands': dict with 'Left', 'Right', 'DoubleHand' active gestures
                - 'pose': list of active pose gestures
                - 'body_language': list of active body language indicators
        """
        hands = hands_data if hands_data is not None else []
        pose = pose_data if pose_data is not None else []
        face = face_data if face_data is not None else []

        hand_gestures = recognize_hand_gestures(hands, pose, face)
        pose_gestures = recognize_pose_gestures(pose)
        
        # Evaluate stateful body language with calibration
        body_language = self.body_language_recognizer.process(pose, face, timestamp_ms)

        return {
            'hands': hand_gestures,
            'pose': pose_gestures,
            'body_language': body_language,
            'blink_count': self.body_language_recognizer.blink_count
        }
