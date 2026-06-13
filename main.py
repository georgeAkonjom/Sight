import cv2
import mediapipe as mp
import time
import argparse
import sys
import os
from mediapipe.tasks.python import vision

from detectors.hand.hand_detector import HandDetectorWrapper
from detectors.face.face_detector import FaceDetectorWrapper
from detectors.pose.pose_detector import PoseDetectorWrapper

def draw_semi_transparent_rect(frame, pt1, pt2, color, alpha):
    """Draw a semi-transparent rectangle on the frame using alpha blending."""
    x1, y1 = pt1
    x2, y2 = pt2
    h, w, _ = frame.shape
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    
    sub_img = frame[y1:y2, x1:x2]
    rect_img = sub_img.copy()
    cv2.rectangle(rect_img, (0, 0), (x2 - x1, y2 - y1), color, -1)
    
    cv2.addWeighted(rect_img, alpha, sub_img, 1.0 - alpha, 0, dst=sub_img)


def draw_gesture_overlay(frame, gesture_results):
    h, w, _ = frame.shape
    overlay_w = 380
    
    if w < overlay_w + 50:
        overlay_w = int(w * 0.5)

    # Floating card with height 300px in the top right corner
    card_bottom = min(h - 10, 310)

    # Draw semi-transparent background box
    draw_semi_transparent_rect(frame, (w - overlay_w - 10, 10), (w - 10, card_bottom), (20, 20, 20), 0.65)
    cv2.rectangle(frame, (w - overlay_w - 10, 10), (w - 10, card_bottom), (0, 255, 255), 1)
    
    cv2.putText(frame, "SIGHT GESTURE HUB", (w - overlay_w + 5, 35), 
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 1)
    
    y_pos = 70
    
    # 1. Hand Gestures
    cv2.putText(frame, "HAND GESTURES", (w - overlay_w + 10, y_pos), 
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0), 1)
    y_pos += 20
    
    hands = gesture_results.get('hands', {})
    left_gestures = hands.get('Left', [])
    right_gestures = hands.get('Right', [])
    double_gestures = hands.get('DoubleHand', [])
    
    has_hands = False
    if left_gestures:
        has_hands = True
        cv2.putText(frame, f"Left: {', '.join(left_gestures[:2])}", (w - overlay_w + 20, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        y_pos += 18
    if right_gestures:
        has_hands = True
        cv2.putText(frame, f"Right: {', '.join(right_gestures[:2])}", (w - overlay_w + 20, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        y_pos += 18
    if double_gestures:
        has_hands = True
        cv2.putText(frame, f"Double: {', '.join(double_gestures)}", (w - overlay_w + 20, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
        y_pos += 18
        
    if not has_hands:
        cv2.putText(frame, "None detected", (w - overlay_w + 20, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
        y_pos += 18
        
    # 2. Body Language
    y_pos += 15
    blink_count = gesture_results.get('blink_count', 0)
    cv2.putText(frame, f"BODY LANGUAGE (Blinks: {blink_count})", (w - overlay_w + 10, y_pos), 
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)
    y_pos += 20
    
    body_lang = gesture_results.get('body_language', [])
    if body_lang:
        for bl in body_lang[:5]:
            if y_pos + 18 <= card_bottom:
                cv2.putText(frame, f"- {bl}", (w - overlay_w + 20, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)
                y_pos += 18
    else:
        if y_pos + 18 <= card_bottom:
            cv2.putText(frame, "Neutral / Relaxed", (w - overlay_w + 20, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
            y_pos += 18

def main():
    parser = argparse.ArgumentParser(description="Multi-Module Hand, Face, and Pose Tracker")
    parser.add_argument('--hand', action='store_true', help='Enable Hand Tracking')
    parser.add_argument('--face', action='store_true', help='Enable Face Tracking')
    parser.add_argument('--pose', action='store_true', help='Enable Pose Tracking')
    parser.add_argument('--all', action='store_true', help='Enable all modules')
    parser.add_argument('--gesture', action='store_true', help='Enable Gesture and Body Language Recognition')
    parser.add_argument('--input', type=str, help='Path to input video file, webcam index, or IP URL')
    parser.add_argument('--url', type=str, help='IP Webcam URL (deprecated: use --input instead)')
    parser.add_argument('--output', type=str, help='Path to save the output annotated video (optional)')
    parser.add_argument('--no-flip', action='store_true', help='Disable horizontal flipping of the frame')
    parser.add_argument('--mode', type=str, choices=['live', 'video'], help='Force running mode: "live" (LIVE_STREAM) or "video" (VIDEO)')
    parser.add_argument('--headless', action='store_true', help='Run without displaying the output window')
    
    args = parser.parse_args()

    # If gesture recognition is enabled and no specific detector is specified, default to --all
    if args.gesture and not (args.hand or args.face or args.pose or args.all):
        print("Gesture recognition enabled. Activating all tracker modules (hand, face, pose).")
        args.all = True

    # If no flags are provided, default to --hand
    if not (args.hand or args.face or args.pose or args.all or args.gesture):
        print("No module specified. Defaulting to Hand Tracking.")
        args.hand = True

    # Determine Video Source
    if args.input is not None:
        video_source = args.input
    elif args.url is not None:
        video_source = args.url
    else:
        video_source = 0

    # If the video source is a digit string, cast it to an integer device index
    if isinstance(video_source, str) and video_source.isdigit():
        video_source = int(video_source)
    # Common fix: IP Webcam usually requires /video suffix for the raw stream
    elif isinstance(video_source, str) and video_source.startswith("http") and not video_source.endswith("/video"):
        if not video_source.endswith("/"):
            video_source += "/"
        video_source += "video"

    # Detect if source is a file on disk
    is_file = False
    if isinstance(video_source, str):
        is_url = any(video_source.startswith(proto) for proto in ["http://", "https://", "rtsp://", "rtmp://"])
        if not is_url and os.path.exists(video_source):
            is_file = True

    # Determine running mode
    if args.mode == 'video':
        running_mode = vision.RunningMode.VIDEO
    elif args.mode == 'live':
        running_mode = vision.RunningMode.LIVE_STREAM
    else:
        # Auto-detect: default to VIDEO mode for files, LIVE_STREAM mode otherwise
        running_mode = vision.RunningMode.VIDEO if is_file else vision.RunningMode.LIVE_STREAM

    active_detectors = []
    
    if args.hand or args.all:
        active_detectors.append(HandDetectorWrapper(running_mode=running_mode))
    if args.face or args.all:
        active_detectors.append(FaceDetectorWrapper(running_mode=running_mode))
    if args.pose or args.all:
        active_detectors.append(PoseDetectorWrapper(running_mode=running_mode))

    gesture_evaluator = None
    if args.gesture:
        from gesture_recognition.evaluator import GestureEvaluator
        gesture_evaluator = GestureEvaluator()

    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        sys.exit(1)
        
    # Minimize buffering latency only for real-time camera feeds
    if running_mode == vision.RunningMode.LIVE_STREAM:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    out = None
    if args.output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"Saving output video to: {args.output}")
        
    mode_str = "VIDEO (Synchronous)" if running_mode == vision.RunningMode.VIDEO else "LIVE_STREAM (Asynchronous)"
    print(f"Streaming from: {'Local Camera' if video_source == 0 else video_source}")
    print(f"Running mode: {mode_str}")
    if not args.headless:
        print("Press 'q' to quit.")

    frame_count = 0
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                if running_mode == vision.RunningMode.VIDEO:
                    print("Reached end of video file.")
                    break
                continue
                
            # Flip for natural mirrored view (only default to True for live stream when not disabled)
            should_flip = not args.no_flip and running_mode == vision.RunningMode.LIVE_STREAM
            if should_flip:
                frame = cv2.flip(frame, 1)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Compute monotonic timestamp
            if running_mode == vision.RunningMode.VIDEO:
                timestamp_ms = int(frame_count * 1000 / fps)
                frame_count += 1
            else:
                timestamp_ms = int(time.time() * 1000)
            
            # 1. Trigger inference for all active detectors
            for detector in active_detectors:
                detector.process_frame(mp_image, timestamp_ms)
            
            # 2. Draw results from all active detectors
            for detector in active_detectors:
                detector.draw(frame)

            # 3. Perform gesture recognition and overlay it if enabled
            if gesture_evaluator:
                hands_data = []
                pose_data = []
                face_data = []
                for detector in active_detectors:
                    detector_name = detector.__class__.__name__
                    if detector_name == 'HandDetectorWrapper':
                        hands_data = detector.get_latest_data()
                    elif detector_name == 'PoseDetectorWrapper':
                        pose_data = detector.get_latest_data()
                    elif detector_name == 'FaceDetectorWrapper':
                        face_data = detector.get_latest_data()
                
                gesture_results = gesture_evaluator.evaluate(
                    hands_data=hands_data,
                    pose_data=pose_data,
                    face_data=face_data,
                    timestamp_ms=timestamp_ms
                )
                draw_gesture_overlay(frame, gesture_results)

            if out:
                out.write(frame)

            if not args.headless:
                cv2.imshow("Multi-Module Tracker", frame)
                
                # Pacing for file streams; minimal delay for live camera
                delay = max(1, int(1000 / fps)) if running_mode == vision.RunningMode.VIDEO else 1
                if cv2.waitKey(delay) & 0xFF == ord('q'):
                    break

    finally:
        print("\nCleaning up...")
        for detector in active_detectors:
            detector.close()
        if out:
            out.release()
            print("Output video closed and saved.")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
