import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from detectors.base_detector import BaseDetector
from helpers.coordinate_translator import CoordinateTranslator

class FaceDetectorWrapper(BaseDetector):
    def __init__(self, model_path='models/face_landmarker.task', running_mode=vision.RunningMode.LIVE_STREAM):
        super().__init__(model_path, running_mode)
        
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=self.model_path),
            running_mode=self.running_mode,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            result_callback=self._result_callback if self.running_mode == vision.RunningMode.LIVE_STREAM else None
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        self.translator = CoordinateTranslator()
        
        # Rendering utilities
        self.mp_drawing = mp.tasks.vision.drawing_utils
        self.mp_drawing_styles = mp.tasks.vision.drawing_styles
        self.mp_face_mesh = mp.tasks.vision.FaceLandmarksConnections

    def get_latest_data(self):
        if not self.latest_result or not self.latest_result.face_landmarks:
            return []

        faces_data = []
        for idx, face_landmarks in enumerate(self.latest_result.face_landmarks):
            # Parse landmarks using CoordinateTranslator
            landmarks = self.translator.get_landmarks(face_landmarks)
            
            # Blendshapes extraction
            blendshapes = []
            if self.latest_result.face_blendshapes and idx < len(self.latest_result.face_blendshapes):
                for category in self.latest_result.face_blendshapes[idx]:
                    blendshapes.append({
                        'category_name': category.category_name,
                        'score': category.score
                    })
                    
            # Transformation matrix extraction
            transformation_matrix = None
            if self.latest_result.facial_transformation_matrixes and idx < len(self.latest_result.facial_transformation_matrixes):
                transformation_matrix = self.latest_result.facial_transformation_matrixes[idx].tolist()

            faces_data.append({
                'landmarks': landmarks,
                'blendshapes': blendshapes,
                'transformation_matrix': transformation_matrix,
                'raw_landmarks': face_landmarks
            })
        return faces_data

    def draw(self, frame):
        faces_data = self.get_latest_data()
        if not faces_data:
            return

        for face in faces_data:
            # Draw face mesh
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face['raw_landmarks'],
                connections=self.mp_face_mesh.FACE_LANDMARKS_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face['raw_landmarks'],
                connections=self.mp_face_mesh.FACE_LANDMARKS_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face['raw_landmarks'],
                connections=self.mp_face_mesh.FACE_LANDMARKS_RIGHT_IRIS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style()
            )
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face['raw_landmarks'],
                connections=self.mp_face_mesh.FACE_LANDMARKS_LEFT_IRIS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style()
            )
