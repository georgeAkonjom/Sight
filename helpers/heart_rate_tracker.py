import cv2
import numpy as np
import math

class HeartRateTracker:
    def __init__(self, buffer_size=150):
        self.buffer_size = buffer_size
        self.rgb_buffer = []  # List of mean (R, G, B) values
        self.bvp_history = []  # For plotting the moving PPG graph
        self.bpm = 0.0
        self.hrv = 0.0
        self.calibrated = False

    def get_roi_mean(self, frame, landmarks):
        """
        Extract the mean RGB values from Forehead and Left/Right Cheek ROIs.
        """
        h, w, _ = frame.shape
        
        # Face scale index: 33 (L eye outer), 263 (R eye outer)
        # We calculate the 2D distance between these points
        scale = math.sqrt((landmarks[33]['x'] - landmarks[263]['x'])**2 + 
                          (landmarks[33]['y'] - landmarks[263]['y'])**2) * w
        if scale == 0:
            scale = 100.0

        # Define ROI coordinates (centers)
        # Forehead=9, L Cheek=117, R Cheek=347
        rois = [
            {'center': landmarks[9],   'w_scale': 0.40, 'h_scale': 0.15},  # Forehead
            {'center': landmarks[117], 'w_scale': 0.20, 'h_scale': 0.20},  # Left Cheek
            {'center': landmarks[347], 'w_scale': 0.20, 'h_scale': 0.20}   # Right Cheek
        ]

        r_means, g_means, b_means = [], [], []

        for roi in rois:
            cx = int(roi['center']['x'] * w)
            cy = int(roi['center']['y'] * h)
            rw = int(scale * roi['w_scale'])
            rh = int(scale * roi['h_scale'])

            # Bounding box
            x1 = max(0, cx - rw // 2)
            y1 = max(0, cy - rh // 2)
            x2 = min(w, cx + rw // 2)
            y2 = min(h, cy + rh // 2)

            if x2 > x1 and y2 > y1:
                roi_pixels = frame[y1:y2, x1:x2]
                # OpenCV reads in BGR format
                b_mean = np.mean(roi_pixels[:, :, 0])
                g_mean = np.mean(roi_pixels[:, :, 1])
                r_mean = np.mean(roi_pixels[:, :, 2])
                
                b_means.append(b_mean)
                g_means.append(g_mean)
                r_means.append(r_mean)

        if len(r_means) == 0:
            return None

        # Return the overall average of all active ROIs
        return np.mean(r_means), np.mean(g_means), np.mean(b_means)

    def update(self, frame, landmarks, fps=30.0):
        """
        Process the frame, update the signal buffer, and compute HR and HRV if buffer is full.
        """
        # 1. Get raw color signals
        rgb = self.get_roi_mean(frame, landmarks)
        if rgb is None:
            return False

        self.rgb_buffer.append(rgb)
        
        # Keep buffer to sliding window size
        if len(self.rgb_buffer) > self.buffer_size:
            self.rgb_buffer.pop(0)

        # 2. Analyze signal once buffer is full
        if len(self.rgb_buffer) >= self.buffer_size:
            self.process_signal(fps)
            self.calibrated = True
        return True

    def process_signal(self, fps):
        """
        Run POS algorithm, FFT-based bandpass filtering, and spectral analysis for HR/HRV.
        """
        H_raw = np.array(self.rgb_buffer)

        # A. Normalization
        mean_rgb = np.mean(H_raw, axis=0)
        H_norm = H_raw / mean_rgb

        # B. POS (Plane-Orthogonal-to-Skin)
        # X = G - B
        # Y = -2*R + G + B
        X = H_norm[:, 1] - H_norm[:, 2]
        Y = -2.0 * H_norm[:, 0] + H_norm[:, 1] + H_norm[:, 2]

        std_X = np.std(X)
        std_Y = np.std(Y)
        if std_Y == 0: std_Y = 0.001
        alpha = std_X / std_Y

        bvp = X - alpha * Y

        # C. FFT-based Ideal Bandpass Filter [0.75, 3.0] Hz (45 to 180 bpm)
        fft_sig = np.fft.rfft(bvp)
        freqs = np.fft.rfftfreq(len(bvp), d=1.0/fps)
        fft_sig[(freqs < 0.75) | (freqs > 3.0)] = 0.0
        filtered_bvp = np.fft.irfft(fft_sig, n=len(bvp))

        # Normalize BVP amplitude for display
        std_val = np.std(filtered_bvp)
        if std_val > 0:
            filtered_bvp = filtered_bvp / std_val

        # Store filtered value for waveform plotting
        self.bvp_history.append(filtered_bvp[-1])
        if len(self.bvp_history) > 100:
            self.bvp_history.pop(0)

        # D. High-resolution Frequency Estimation (Zero-Padding)
        padded_len = 2048
        fft_sig_padded = np.fft.rfft(filtered_bvp, n=padded_len)
        freqs_padded = np.fft.rfftfreq(padded_len, d=1.0/fps)

        mask = (freqs_padded >= 0.75) & (freqs_padded <= 3.0)
        valid_freqs = freqs_padded[mask]
        valid_power = np.abs(fft_sig_padded[mask]) ** 2

        if len(valid_power) > 0:
            peak_idx = np.argmax(valid_power)
            self.bpm = valid_freqs[peak_idx] * 60.0
        else:
            self.bpm = 0.0

        # E. Peak Detection & HRV (RMSSD)
        # Peaks are local maxima > 0.0
        peaks = []
        min_dist = int(fps * 0.4)  # Refractory period corresponding to max 150 bpm
        last_peak_idx = -min_dist

        for i in range(1, len(filtered_bvp) - 1):
            if filtered_bvp[i] > filtered_bvp[i-1] and filtered_bvp[i] > filtered_bvp[i+1]:
                if filtered_bvp[i] > 0.2:  # Threshold
                    if i - last_peak_idx >= min_dist:
                        peaks.append(i)
                        last_peak_idx = i

        # Calculate RMSSD
        peak_times_ms = [p * 1000.0 / fps for p in peaks]
        ibis = [peak_times_ms[i+1] - peak_times_ms[i] for i in range(len(peak_times_ms) - 1)]

        if len(ibis) > 1:
            diffs = [ibis[i+1] - ibis[i] for i in range(len(ibis) - 1)]
            self.hrv = math.sqrt(np.mean([d**2 for d in diffs]))
        else:
            self.hrv = 0.0

    def draw_ppg_graph(self, frame, x_start=20, y_center=150, width=200, height=60):
        """
        Draw a scrolling PPG pulse waveform graph directly on the OpenCV frame.
        """
        if len(self.bvp_history) < 2:
            return

        # Draw background container
        sub_img = frame[y_center - height//2 : y_center + height//2, x_start : x_start + width]
        rect_img = sub_img.copy()
        cv2.rectangle(rect_img, (0, 0), (width, height), (30, 30, 30), -1)
        cv2.addWeighted(rect_img, 0.6, sub_img, 0.4, 0, dst=sub_img)
        cv2.rectangle(frame, (x_start, y_center - height//2), (x_start + width, y_center + height//2), (100, 100, 100), 1)

        # Plot waveform lines
        num_points = len(self.bvp_history)
        step_x = width / 100.0
        
        points = []
        for idx, val in enumerate(self.bvp_history):
            # Scale BVP values (standardized around 0, peaks range [-2.5, 2.5])
            px = int(x_start + idx * step_x)
            # Invert y since cv2 origin is top-left
            py = int(y_center - (val * (height * 0.4) / 2.5))
            # Clip y boundaries
            py = max(y_center - height//2 + 3, min(y_center + height//2 - 3, py))
            points.append((px, py))

        for idx in range(len(points) - 1):
            cv2.line(frame, points[idx], points[idx+1], (0, 255, 0), 1, cv2.LINE_AA)

        cv2.putText(frame, "PPG WAVEFORM", (x_start + 5, y_center - height//2 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)

if __name__ == "__main__":
    print("Testing HeartRateTracker with synthetic BVP signal...")
    tracker = HeartRateTracker(buffer_size=150)
    
    # Simulate a face landmark dictionary with minimal landmarks for testing
    # We only need 9, 117, 347, 33, 263
    mock_landmarks = {
        9: {'x': 0.5, 'y': 0.25},
        117: {'x': 0.42, 'y': 0.5},
        347: {'x': 0.58, 'y': 0.5},
        33: {'x': 0.4, 'y': 0.4},
        263: {'x': 0.6, 'y': 0.4}
    }
    
    fps = 30.0
    heart_rate_hz = 1.25  # 1.25 Hz * 60 = 75 bpm
    
    # We create a dummy frame (BGR image)
    frame_size = (480, 640, 3)
    
    print("Streaming synthetic frames to buffer...")
    for frame_idx in range(150):
        t = frame_idx / fps
        pulse_sin = math.sin(2 * math.pi * heart_rate_hz * t)
        pulse_cos = math.cos(2 * math.pi * heart_rate_hz * t)
        
        # Skin baseline value (e.g. R=180, G=130, B=110)
        # Shift phases to prevent POS from canceling out in-phase signals
        r_val = 180
        g_val = int(130 + 12.0 * pulse_sin)
        b_val = int(110 + 6.0 * pulse_cos)
        
        # Create solid color dummy frame
        frame = np.zeros(frame_size, dtype=np.uint8)
        frame[:, :, 0] = b_val
        frame[:, :, 1] = g_val
        frame[:, :, 2] = r_val
        
        success = tracker.update(frame, mock_landmarks, fps=fps)
        assert success, "Failed to update tracker"

    print(f"Calculated Heart Rate: {tracker.bpm:.2f} BPM")
    print(f"Calculated HRV (RMSSD): {tracker.hrv:.2f} ms")
    
    # Expected heart rate: ~75 BPM (within a delta of 1.0 BPM due to window resolution)
    assert abs(tracker.bpm - 75.0) < 1.0, f"Expected ~75.0 BPM, got {tracker.bpm:.2f}"
    # Synthetic clean sine wave has a constant interval, but discrete sampling introduces 1-frame jitter (33.3ms)
    assert tracker.hrv < 30.0, f"Expected low HRV on perfect sine wave, got {tracker.hrv:.2f}"
    
    print("HeartRateTracker tests passed successfully!")
