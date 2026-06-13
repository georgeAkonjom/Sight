import cv2
import numpy as np
import math

class GazeTracker:
    def __init__(self):
        # 3D model points of a standard face (in mm)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip (4)
            (0.0, -330.0, -65.0),        # Chin (152)
            (-225.0, 170.0, -135.0),     # Left eye outer corner (33)
            (225.0, 170.0, -135.0),      # Right eye outer corner (263)
            (-150.0, -150.0, -125.0),    # Left mouth corner (61)
            (150.0, -150.0, -125.0)      # Right mouth corner (291)
        ], dtype=np.float32)

        # Calibration state
        self.calibration_data = []  # List of tuples: (feature_row, target_x, target_y)
        self.W_x = None  # Weights for mapping features to screen X
        self.W_y = None  # Weights for mapping features to screen Y
        self.calibrated = False

    def get_pnp_pose(self, landmarks, frame_size):
        """
        Estimate head pose (rotation and translation vectors) using Perspective-n-Point.
        """
        h, w = frame_size
        
        # Extract corresponding 2D points from MediaPipe face landmarks
        # MediaPipe indices: Nose=4, Chin=152, L Eye Outer=33, R Eye Outer=263, L Mouth=61, R Mouth=291
        indices = [4, 152, 33, 263, 61, 291]
        
        # Face landmarks are normalized coordinates [0.0, 1.0]. Convert to pixel space.
        image_points = []
        for idx in indices:
            lm = landmarks[idx]
            image_points.append((lm['x'] * w, lm['y'] * h))
        
        image_points = np.array(image_points, dtype=np.float32)

        # Approximate camera matrix
        focal_length = w
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)
        
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)  # Assume no lens distortion

        # Solve for pose
        success, rvec, tvec = cv2.solvePnP(
            self.model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return None, None, None, None

        # Calculate rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Calculate Euler angles (pitch, yaw, roll)
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            pitch = math.atan2(R[2, 1], R[2, 2])
            yaw = math.atan2(-R[2, 0], sy)
            roll = math.atan2(R[1, 0], R[0, 0])
        else:
            pitch = math.atan2(-R[1, 2], R[1, 1])
            yaw = math.atan2(-R[2, 0], sy)
            roll = 0.0

        # Convert to degrees
        pitch = math.degrees(pitch)
        yaw = math.degrees(yaw)
        roll = math.degrees(roll)

        return rvec, tvec, (pitch, yaw, roll), camera_matrix

    def get_pupil_offsets(self, landmarks):
        """
        Calculate relative pupil offsets (dx, dy) for left and right eyes.
        Returns normalized coordinates relative to eye width/height to be scale-invariant.
        """
        if len(landmarks) < 478:
            return 0.0, 0.0, 0.0, 0.0

        # Left Eye: outer corner=33, inner corner=133, pupil center=468
        l_outer = landmarks[33]
        l_inner = landmarks[133]
        l_pupil = landmarks[468]

        # Left eye width in normalized coords
        l_width = math.sqrt((l_outer['x'] - l_inner['x'])**2 + (l_outer['y'] - l_inner['y'])**2)
        if l_width == 0: l_width = 0.001
        
        l_center_x = (l_outer['x'] + l_inner['x']) / 2.0
        l_center_y = (l_outer['y'] + l_inner['y']) / 2.0

        dx_l = (l_pupil['x'] - l_center_x) / l_width
        dy_l = (l_pupil['y'] - l_center_y) / l_width

        # Right Eye: outer corner=263, inner corner=362, pupil center=473
        r_outer = landmarks[263]
        r_inner = landmarks[362]
        r_pupil = landmarks[473]

        # Right eye width
        r_width = math.sqrt((r_outer['x'] - r_inner['x'])**2 + (r_outer['y'] - r_inner['y'])**2)
        if r_width == 0: r_width = 0.001

        r_center_x = (r_outer['x'] + r_inner['x']) / 2.0
        r_center_y = (r_outer['y'] + r_inner['y']) / 2.0

        dx_r = (r_pupil['x'] - r_center_x) / r_width
        dy_r = (r_pupil['y'] - r_center_y) / r_width

        return dx_l, dy_l, dx_r, dy_r

    def construct_features(self, angles, pupil_offsets):
        """
        Construct a non-linear feature row for regression.
        Combines head angles (pitch, yaw) and pupil displacements, including interaction terms.
        """
        pitch, yaw, roll = angles
        dx_l, dy_l, dx_r, dy_r = pupil_offsets

        # Averages of eyes
        dx_avg = (dx_l + dx_r) / 2.0
        dy_avg = (dy_l + dy_r) / 2.0

        # Feature vector components:
        # 1. Intercept
        # 2-3. Head rotation (yaw, pitch)
        # 4-5. Relative pupil offsets (dx, dy)
        # 6-7. Interaction terms (yaw * dx, pitch * dy) - compensates for compensatory head-eye coordination
        feature_row = [
            1.0,
            yaw,
            pitch,
            dx_avg,
            dy_avg,
            yaw * dx_avg,
            pitch * dy_avg
        ]
        return np.array(feature_row, dtype=np.float32)

    def record_calibration_frame(self, landmarks, frame_size, target_x, target_y):
        """
        Collect gaze features for a specific screen target.
        """
        rvec, tvec, angles, _ = self.get_pnp_pose(landmarks, frame_size)
        if angles is None:
            return False

        pupil_offsets = self.get_pupil_offsets(landmarks)
        feature_row = self.construct_features(angles, pupil_offsets)

        self.calibration_data.append((feature_row, target_x, target_y))
        return True

    def train(self):
        """
        Solve linear least-squares regression to map feature rows to screen coordinates.
        """
        if len(self.calibration_data) < 5:
            print("[GazeTracker] Warning: Not enough calibration data. Need at least 5 points.")
            return False

        X = []
        Y_x = []
        Y_y = []

        for row, tx, ty in self.calibration_data:
            X.append(row)
            Y_x.append(tx)
            Y_y.append(ty)

        X = np.array(X, dtype=np.float32)
        Y_x = np.array(Y_x, dtype=np.float32)
        Y_y = np.array(Y_y, dtype=np.float32)

        # Solve regression: X * W = Y
        # W = (X^T * X)^(-1) * X^T * Y
        self.W_x, _, _, _ = np.linalg.lstsq(X, Y_x, rcond=None)
        self.W_y, _, _, _ = np.linalg.lstsq(X, Y_y, rcond=None)

        self.calibrated = True
        print(f"[GazeTracker] Calibration complete. Model fitted on {len(self.calibration_data)} frames.")
        return True

    def predict(self, landmarks, frame_size):
        """
        Predict screen coordinates (X, Y) based on current landmarks.
        """
        if not self.calibrated or self.W_x is None or self.W_y is None:
            return None

        rvec, tvec, angles, _ = self.get_pnp_pose(landmarks, frame_size)
        if angles is None:
            return None

        pupil_offsets = self.get_pupil_offsets(landmarks)
        feature_row = self.construct_features(angles, pupil_offsets)

        screen_x = float(np.dot(feature_row, self.W_x))
        screen_y = float(np.dot(feature_row, self.W_y))

        return screen_x, screen_y

    def draw_gaze_debug(self, frame, landmarks):
        """
        Draw debug visuals on the frame:
        - 3D pose coordinate axes projecting from the nose tip.
        - Text indicators for yaw, pitch, and relative eye offsets.
        """
        h, w, _ = frame.shape
        rvec, tvec, angles, camera_matrix = self.get_pnp_pose(landmarks, (h, w))
        
        if angles is None:
            return

        pitch, yaw, roll = angles
        dx_l, dy_l, dx_r, dy_r = self.get_pupil_offsets(landmarks)

        # 1. Project 3D axes from origin (Nose tip)
        axis_length = 80.0  # in mm
        axis_points = np.array([
            (axis_length, 0.0, 0.0),   # X axis (Red)
            (0.0, axis_length, 0.0),   # Y axis (Green)
            (0.0, 0.0, axis_length)    # Z axis (Blue)
        ], dtype=np.float32)

        dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        projected_points, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
        
        # Nose tip coordinates in pixel space
        nose_lm = landmarks[4]
        origin = (int(nose_lm['x'] * w), int(nose_lm['y'] * h))

        # Projected axis endpoints
        p_x = (int(projected_points[0][0][0]), int(projected_points[0][0][1]))
        p_y = (int(projected_points[1][0][0]), int(projected_points[1][0][1]))
        p_z = (int(projected_points[2][0][0]), int(projected_points[2][0][1]))

        # Draw axis lines
        cv2.line(frame, origin, p_x, (0, 0, 255), 2)  # X axis: Red
        cv2.line(frame, origin, p_y, (0, 255, 0), 2)  # Y axis: Green
        cv2.line(frame, origin, p_z, (255, 0, 0), 2)  # Z axis: Blue (Z points out)

        # 2. Draw text indicators
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        color = (255, 255, 255)
        thickness = 1
        
        cv2.putText(frame, f"Head Yaw: {yaw:.1f} deg", (20, h - y_offset * 3), font, scale, color, thickness)
        cv2.putText(frame, f"Head Pitch: {pitch:.1f} deg", (20, h - y_offset * 2), font, scale, color, thickness)
        cv2.putText(frame, f"Pupil Offset (L): ({dx_l:.2f}, {dy_l:.2f})", (20, h - y_offset * 1), font, scale, color, thickness)
