# coordinate_translator.py

class CoordinateTranslator:
    def get_landmarks(self, hand_landmarks):
        """
        Translates MediaPipe landmarks into a list of dictionaries 
        containing x, y, and z coordinates for all 21 points.
        """
        coords = []
        for lm in hand_landmarks:
            coords.append({
                "x": lm.x,
                "y": lm.y,
                "z": lm.z
            })
        return coords

    def get_structured_data(self, hand_landmarks, handedness):
        """
        Returns a structured dictionary with handedness and landmark coordinates.
        """
        return {
            "handedness": handedness,
            "landmarks": self.get_landmarks(hand_landmarks)
        }
