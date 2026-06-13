import time
from gesture_recognition.utils import dist3d

class BodyLanguageRecognizer:
    """
    Stateful recognizer that runs a calibration process for the first 10 seconds of streaming
    to establish the user's neutral face blendshapes and shoulder posture baseline.
    All subsequent body language is evaluated relative to these customized baselines.
    Also tracks blink counts and pupil stability post-calibration.
    """
    def __init__(self, calibration_duration_ms=10000):
        self.calibration_duration_ms = calibration_duration_ms
        self.start_time_ms = None
        self.is_calibrated = False
        
        # Lists to store calibration samples
        self.face_samples = []
        self.pose_samples = []
        
        # Baseline templates
        self.baselines = {
            'blendshapes': {},
            'l_shoulder_ear': None,
            'r_shoulder_ear': None,
            'shoulder_width': None
        }

        # Blink counter state
        self.blink_count = 0
        self.eyes_closed = False
        self.last_blink_time_ms = None

        # Pupil tracking history (sliding 2-second buffer)
        # Each entry: {'timestamp_ms': t, 'l_rel': (x, y), 'r_rel': (x, y)}
        self.pupil_history = []

        # Arm tracking history (sliding 4-second buffer)
        self.arm_history = []

        # Blink timestamps history for rate analysis (sliding 6-second window)
        self.blink_timestamps = []

    def process(self, pose_data=None, face_data=None, timestamp_ms=None):
        """
        Processes current frame pose and face data.
        During the calibration phase, accumulates features.
        After calibration, checks current parameters against baseline.
        """
        if timestamp_ms is None:
            # Fallback if timestamp not provided
            timestamp_ms = int(time.time() * 1000)
            
        if self.start_time_ms is None:
            self.start_time_ms = timestamp_ms

        elapsed_ms = timestamp_ms - self.start_time_ms
        
        # Extract face blendshapes and landmarks
        blendshapes = {}
        face_landmarks = None
        if face_data and len(face_data) > 0:
            face_landmarks = face_data[0].get('landmarks', [])
            for b in face_data[0].get('blendshapes', []):
                blendshapes[b['category_name']] = b['score']
                
        pose_landmarks = None
        if pose_data and len(pose_data) > 0:
            pose_landmarks = pose_data[0]['landmarks']
            
            # Record arm history (4-second window)
            l_wrist_p = pose_landmarks[15]
            l_elbow_p = pose_landmarks[13]
            r_wrist_p = pose_landmarks[16]
            r_elbow_p = pose_landmarks[14]
            self.arm_history.append({
                'timestamp_ms': timestamp_ms,
                'l_wrist': (l_wrist_p['x'], l_wrist_p['y']),
                'l_elbow': (l_elbow_p['x'], l_elbow_p['y']),
                'r_wrist': (r_wrist_p['x'], r_wrist_p['y']),
                'r_elbow': (r_elbow_p['x'], r_elbow_p['y'])
            })
            
        # Clean history older than 4 seconds (4000 ms)
        self.arm_history = [s for s in self.arm_history if timestamp_ms - s['timestamp_ms'] <= 4000]

        # ----------------------------------------------------
        # 1. Calibration Phase
        # ----------------------------------------------------
        if elapsed_ms < self.calibration_duration_ms:
            if blendshapes:
                self.face_samples.append(blendshapes)
            if pose_landmarks:
                l_shoulder = pose_landmarks[11]
                r_shoulder = pose_landmarks[12]
                l_ear = pose_landmarks[7]
                r_ear = pose_landmarks[8]
                s_width = dist3d(l_shoulder, r_shoulder)
                self.pose_samples.append({
                    'l_dist': l_shoulder['y'] - l_ear['y'],
                    'r_dist': r_shoulder['y'] - r_ear['y'],
                    's_width': s_width
                })
            
            progress = int((elapsed_ms / self.calibration_duration_ms) * 100)
            return [f"Calibrating... {progress}% done"]

        # ----------------------------------------------------
        # 2. Finalize Calibration Baselines
        # ----------------------------------------------------
        if not self.is_calibrated:
            if self.face_samples:
                all_keys = self.face_samples[0].keys()
                for key in all_keys:
                    scores = [sample[key] for sample in self.face_samples if key in sample]
                    self.baselines['blendshapes'][key] = sum(scores) / len(scores) if scores else 0.0
            
            if self.pose_samples:
                l_dists = [s['l_dist'] for s in self.pose_samples]
                r_dists = [s['r_dist'] for s in self.pose_samples]
                s_widths = [s['s_width'] for s in self.pose_samples]
                self.baselines['l_shoulder_ear'] = sum(l_dists) / len(l_dists) if l_dists else 0.4
                self.baselines['r_shoulder_ear'] = sum(r_dists) / len(r_dists) if r_dists else 0.4
                self.baselines['shoulder_width'] = sum(s_widths) / len(s_widths) if s_widths else 0.2
                
            self.is_calibrated = True
            print("[BodyLanguage] 10s calibration complete. Baseline established.")

        # ----------------------------------------------------
        # 3. Active Body Language & Blink Detection Phase
        # ----------------------------------------------------
        body_lang = []
        
        # Helper to compute positive deviation from neutral baseline
        def get_deviation(category):
            val = blendshapes.get(category, 0.0)
            base = self.baselines['blendshapes'].get(category, 0.0)
            return max(0.0, val - base)

        # A. Blink Detection State Machine
        blink_l = blendshapes.get('eyeBlinkLeft', 0.0)
        blink_r = blendshapes.get('eyeBlinkRight', 0.0)
        base_l = self.baselines['blendshapes'].get('eyeBlinkLeft', 0.1)
        base_r = self.baselines['blendshapes'].get('eyeBlinkRight', 0.1)
        
        closed_thresh_l = min(0.85, base_l + 0.45)
        closed_thresh_r = min(0.85, base_r + 0.45)
        open_thresh_l = base_l + 0.2
        open_thresh_r = base_r + 0.2
        
        if blink_l > closed_thresh_l and blink_r > closed_thresh_r:
            self.eyes_closed = True
        elif self.eyes_closed and blink_l < open_thresh_l and blink_r < open_thresh_r:
            self.blink_count += 1
            self.eyes_closed = False
            self.last_blink_time_ms = timestamp_ms
            self.blink_timestamps.append(timestamp_ms)
            
        # Display feedback if a blink was just registered (lasts for 800ms)
        if self.last_blink_time_ms and (timestamp_ms - self.last_blink_time_ms < 800):
            body_lang.append("Blink Registered! (Counted)")

        # B. Pupil Tracking (Sliding 2-second Gaze Stability)
        if face_landmarks and len(face_landmarks) >= 478:
            # Only track pupils when eyes are open to avoid blink artifacts
            if blink_l < closed_thresh_l and blink_r < closed_thresh_r:
                # Left eye corners: 33 (outer), 133 (inner). Left pupil: 468 (iris center)
                l_outer = face_landmarks[33]
                l_inner = face_landmarks[133]
                l_pupil = face_landmarks[468]
                l_mid_x = (l_outer['x'] + l_inner['x']) / 2
                l_mid_y = (l_outer['y'] + l_inner['y']) / 2
                l_rel = (l_pupil['x'] - l_mid_x, l_pupil['y'] - l_mid_y)

                # Right eye corners: 362 (inner), 263 (outer). Right pupil: 473 (iris center)
                r_inner = face_landmarks[362]
                r_outer = face_landmarks[263]
                r_pupil = face_landmarks[473]
                r_mid_x = (r_inner['x'] + r_outer['x']) / 2
                r_mid_y = (r_inner['y'] + r_outer['y']) / 2
                r_rel = (r_pupil['x'] - r_mid_x, r_pupil['y'] - r_mid_y)

                self.pupil_history.append({
                    'timestamp_ms': timestamp_ms,
                    'l_rel': l_rel,
                    'r_rel': r_rel
                })

        # Purge pupil history older than 2 seconds (2000 ms)
        self.pupil_history = [s for s in self.pupil_history if timestamp_ms - s['timestamp_ms'] <= 2000]

        # Purge blink history older than 6 seconds (6000 ms)
        self.blink_timestamps = [t for t in self.blink_timestamps if timestamp_ms - t <= 6000]

        has_shifty_gaze = False

        # 1. Check for Irregular/Rapid Blinks (3 or more blinks in 6 seconds)
        if len(self.blink_timestamps) >= 3:
            body_lang.append("Rapid/Irregular Blinking (Tension)")
            has_shifty_gaze = True

        # Calculate gaze stability standard deviation
        if len(self.pupil_history) >= 10:
            l_xs = [s['l_rel'][0] for s in self.pupil_history]
            l_ys = [s['l_rel'][1] for s in self.pupil_history]
            r_xs = [s['r_rel'][0] for s in self.pupil_history]
            r_ys = [s['r_rel'][1] for s in self.pupil_history]

            def std_dev(vals):
                mean = sum(vals) / len(vals)
                variance = sum((x - mean) ** 2 for x in vals) / len(vals)
                return variance ** 0.5

            std_l_x = std_dev(l_xs)
            std_l_y = std_dev(l_ys)
            std_r_x = std_dev(r_xs)
            std_r_y = std_dev(r_ys)

            # Average pupil dispersion across both eyes
            gaze_dispersion = (std_l_x + std_l_y + std_r_x + std_r_y) / 4

            # 2. Check for Rapid Eye Movements (Saccades) in the last 2 seconds
            saccades = 0
            for idx in range(1, len(self.pupil_history)):
                s_prev = self.pupil_history[idx - 1]
                s_curr = self.pupil_history[idx]
                
                # Compute 2D displacement
                dx = s_curr['l_rel'][0] - s_prev['l_rel'][0]
                dy = s_curr['l_rel'][1] - s_prev['l_rel'][1]
                dist_l = (dx**2 + dy**2)**0.5
                
                # If displacement frame-to-frame exceeds 0.0065, count as a sudden movement
                if dist_l > 0.0065:
                    saccades += 1

            if saccades >= 3:
                body_lang.append("Rapid Eye Movements (Restless)")
                has_shifty_gaze = True

            # Classify gaze stability (triggered by dispersion, rapid blinking, or REMs)
            if gaze_dispersion > 0.0055 or has_shifty_gaze:
                body_lang.append("Shifty Gaze (Restless/Anxious)")
            elif gaze_dispersion < 0.0025:
                body_lang.append("Stable Gaze (Focused/Calm)")
            else:
                body_lang.append("Normal Gaze (Stable)")

        # C. Waving Detection (Active movements over 4 seconds)
        if len(self.arm_history) >= 10 and pose_landmarks:
            l_wrist_rel_x = [s['l_wrist'][0] - s['l_elbow'][0] for s in self.arm_history]
            r_wrist_rel_x = [s['r_wrist'][0] - s['r_elbow'][0] for s in self.arm_history]
            
            def count_swing_oscillations(seq):
                if len(seq) < 5:
                    return 0
                extrema = []
                direction = 0 # 1 for up, -1 for down
                last_val = seq[0]
                last_ext_val = seq[0]
                # Threshold of 0.025 represents about 2.5% of horizontal frame width
                swing_threshold = 0.025
                
                for val in seq[1:]:
                    diff = val - last_val
                    if diff > 0.001:
                        if direction == -1:
                            if last_val - last_ext_val < -swing_threshold:
                                extrema.append(last_val)
                                last_ext_val = last_val
                            direction = 1
                        elif direction == 0:
                            direction = 1
                    elif diff < -0.001:
                        if direction == 1:
                            if last_val - last_ext_val > swing_threshold:
                                extrema.append(last_val)
                                last_ext_val = last_val
                            direction = -1
                        elif direction == 0:
                            direction = -1
                    last_val = val
                return len(extrema)

            # Check Right Hand Waving
            r_shoulder = pose_landmarks[12]
            r_wrist = pose_landmarks[16]
            if r_wrist['y'] < r_shoulder['y']: # Hand is raised above shoulder
                r_range_x = max(r_wrist_rel_x) - min(r_wrist_rel_x)
                r_oscillations = count_swing_oscillations(r_wrist_rel_x)
                if r_range_x > 0.065 and r_oscillations >= 3:
                    body_lang.append("Right Hand Waving (Active)")

            # Check Left Hand Waving
            l_shoulder = pose_landmarks[11]
            l_wrist = pose_landmarks[15]
            if l_wrist['y'] < l_shoulder['y']: # Hand is raised above shoulder
                l_range_x = max(l_wrist_rel_x) - min(l_wrist_rel_x)
                l_oscillations = count_swing_oscillations(l_wrist_rel_x)
                if l_range_x > 0.065 and l_oscillations >= 3:
                    body_lang.append("Left Hand Waving (Active)")

        # D. Micro-expressions
        # 1. Face Scrunched (Disgust or deep concentration)
        dev_nose_wrinkle = (get_deviation('noseWrinkleLeft') + get_deviation('noseWrinkleRight')) / 2
        dev_eye_squint = (get_deviation('eyeSquintLeft') + get_deviation('eyeSquintRight')) / 2
        if dev_nose_wrinkle > 0.25 and dev_eye_squint > 0.25:
            body_lang.append("Face Scrunched (Discomfort/Concentration)")

        # 1b. Squinting (Narrowed eyes - only when not fully scrunched)
        elif dev_eye_squint > 0.35:
            body_lang.append("Squinting (Eyes Narrowed)")

        # 2. Smiling (above baseline)
        dev_smile = (get_deviation('mouthSmileLeft') + get_deviation('mouthSmileRight')) / 2
        if dev_smile > 0.35:
            body_lang.append("Smiling (Friendly/Welcoming)")

        # 2b. Smirking (Asymmetric Smile)
        smile_l = blendshapes.get('mouthSmileLeft', 0.0)
        smile_r = blendshapes.get('mouthSmileRight', 0.0)
        base_smile_l = self.baselines['blendshapes'].get('mouthSmileLeft', 0.0)
        base_smile_r = self.baselines['blendshapes'].get('mouthSmileRight', 0.0)
        dev_smile_l = max(0.0, smile_l - base_smile_l)
        dev_smile_r = max(0.0, smile_r - base_smile_r)
        if abs(dev_smile_l - dev_smile_r) > 0.30:
            if dev_smile_l > dev_smile_r and dev_smile_l > 0.25:
                body_lang.append("Smirking (Left Side)")
            elif dev_smile_r > dev_smile_l and dev_smile_r > 0.25:
                body_lang.append("Smirking (Right Side)")

        # 3. Frowning (above baseline)
        dev_frown = (get_deviation('mouthFrownLeft') + get_deviation('mouthFrownRight')) / 2
        dev_brow_down = (get_deviation('browDownLeft') + get_deviation('browDownRight')) / 2
        if dev_frown > 0.25 and dev_brow_down > 0.2:
            body_lang.append("Frowning (Disappointed/Concerned)")

        # 4. Surprised / Shocked (above baseline)
        dev_jaw_open = get_deviation('jawOpen')
        dev_eye_wide = (get_deviation('eyeWideLeft') + get_deviation('eyeWideRight')) / 2
        if dev_jaw_open > 0.35 and dev_eye_wide > 0.2:
            body_lang.append("Surprised / Shocked")

        # 4b. Mouth Open (Parted Lips - when not surprised or yawning)
        elif dev_jaw_open > 0.28 and dev_eye_wide <= 0.15:
            body_lang.append("Mouth Open (Parted Lips)")

        # 5. Angry / Frustrated / Tense (above baseline)
        dev_mouth_press = (get_deviation('mouthPressLeft') + get_deviation('mouthPressRight')) / 2
        if dev_brow_down > 0.35 and (dev_mouth_press > 0.25 or dev_nose_wrinkle > 0.25):
            body_lang.append("Tense / Frustrated")

        # 6. Winking (check if difference in blink goes above baseline differential)
        diff_l = max(0.0, blink_l - base_l)
        diff_r = max(0.0, blink_r - base_r)
        if abs(diff_l - diff_r) > 0.5:
            if diff_l > diff_r:
                body_lang.append("Winking (Left Eye)")
            else:
                body_lang.append("Winking (Right Eye)")

        # 7. Eyes Closed (only triggers if not just winking or blinking)
        if diff_l > 0.7 and diff_r > 0.7:
            # Check to avoid overlap with a transient blink
            if not (self.last_blink_time_ms and (timestamp_ms - self.last_blink_time_ms < 500)):
                body_lang.append("Eyes Closed")

        # 8. Yawning / Tired
        if dev_jaw_open > 0.45 and dev_eye_squint > 0.35:
            body_lang.append("Yawning / Tired")

        # 9. Puzzled / Thinking
        dev_pucker = get_deviation('mouthPucker')
        dev_brow_inner_up = get_deviation('browInnerUp')
        if dev_pucker > 0.3 or dev_brow_inner_up > 0.3:
            body_lang.append("Puzzled / In Thought")

        # 9b. Eyebrows Raised (Surprise/Questioning)
        dev_brow_outer_up_l = get_deviation('browOuterUpLeft')
        dev_brow_outer_up_r = get_deviation('browOuterUpRight')
        if dev_brow_inner_up > 0.35 or (dev_brow_outer_up_l + dev_brow_outer_up_r)/2 > 0.35:
            body_lang.append("Eyebrows Raised (Surprise/Questioning)")

        # 9c. Skeptical Eyebrow Raise (Asymmetric Raise)
        if abs(dev_brow_outer_up_l - dev_brow_outer_up_r) > 0.30:
            if dev_brow_outer_up_l > dev_brow_outer_up_r and dev_brow_outer_up_l > 0.25:
                body_lang.append("Skeptical Eyebrow Raise (Left)")
            elif dev_brow_outer_up_r > dev_brow_outer_up_l and dev_brow_outer_up_r > 0.25:
                body_lang.append("Skeptical Eyebrow Raise (Right)")

        # 9d. Tongue Out (Playful/Teasing)
        dev_tongue_out = get_deviation('tongueOut')
        if dev_tongue_out > 0.25:
            body_lang.append("Tongue Out (Playful/Teasing)")

        # Pose-based body language
        if pose_landmarks:
            l_shoulder = pose_landmarks[11]
            r_shoulder = pose_landmarks[12]
            l_ear = pose_landmarks[7]
            r_ear = pose_landmarks[8]
            l_wrist = pose_landmarks[15]
            r_wrist = pose_landmarks[16]
            l_elbow = pose_landmarks[13]
            r_elbow = pose_landmarks[14]
            
            s_width = dist3d(l_shoulder, r_shoulder)
            if s_width == 0:
                s_width = 0.001
                
            curr_l_dist = l_shoulder['y'] - l_ear['y']
            curr_r_dist = r_shoulder['y'] - r_ear['y']
            
            base_l_dist = self.baselines['l_shoulder_ear']
            base_r_dist = self.baselines['r_shoulder_ear']
            
            # 10. Shoulders Tight / Shrugging
            if base_l_dist and base_r_dist:
                if curr_l_dist < 0.78 * base_l_dist and curr_r_dist < 0.78 * base_r_dist:
                    body_lang.append("Shoulders Tight / Shrugging (Tension/Uncertainty)")

            # 11. Defensive Posture (Arms Crossed)
            if dist3d(l_wrist, r_elbow) < 0.7 * s_width and dist3d(r_wrist, l_elbow) < 0.7 * s_width:
                if l_wrist['y'] > l_shoulder['y'] and r_wrist['y'] > r_shoulder['y']:
                    body_lang.append("Defensive Posture (Arms Crossed)")

            # 12. Head Tilt
            ear_y_diff = l_ear['y'] - r_ear['y']
            if abs(ear_y_diff) > 0.08 * s_width:
                if ear_y_diff > 0:
                    body_lang.append("Head Tilted Right (Listening/Curious)")
                else:
                    body_lang.append("Head Tilted Left (Listening/Curious)")

            # 13. Closed Posture (Hands Near Hips)
            l_hip = pose_landmarks[23]
            r_hip = pose_landmarks[24]
            if dist3d(l_wrist, l_hip) < 0.35 * s_width and dist3d(r_wrist, r_hip) < 0.35 * s_width:
                if l_wrist['y'] > l_hip['y'] and r_wrist['y'] > r_hip['y']:
                    body_lang.append("Closed Posture (Hands Near Hips)")

        return list(set(body_lang))
