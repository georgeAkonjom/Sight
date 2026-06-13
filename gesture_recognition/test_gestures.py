import sys
import os

# Allow importing from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gesture_recognition.evaluator import GestureEvaluator

def create_mock_landmarks(n):
    """Create n mock landmarks at (0.5, 0.5, 0.0)."""
    return [{'x': 0.5, 'y': 0.5, 'z': 0.0} for _ in range(n)]

def set_eye_landmarks(landmarks, l_pupil_x=0.45, r_pupil_x=0.55):
    """Set coordinates for eye corners and pupils to simulate gazes."""
    # Left eye: corners 33, 133, pupil 468
    landmarks[33] = {'x': 0.43, 'y': 0.3, 'z': 0.0}
    landmarks[133] = {'x': 0.47, 'y': 0.3, 'z': 0.0}
    landmarks[468] = {'x': l_pupil_x, 'y': 0.3, 'z': 0.0}
    
    # Right eye: corners 362, 263, pupil 473
    landmarks[362] = {'x': 0.53, 'y': 0.3, 'z': 0.0}
    landmarks[263] = {'x': 0.57, 'y': 0.3, 'z': 0.0}
    landmarks[473] = {'x': r_pupil_x, 'y': 0.3, 'z': 0.0}

def test_full_gesture_system():
    print("Testing Gesture Recognition with mock data...")
    evaluator = GestureEvaluator()

    # 1. Setup Mock Pose (Hands Up)
    pose_landmarks = create_mock_landmarks(33)
    pose_landmarks[0] = {'x': 0.5, 'y': 0.2, 'z': 0.0}      # Nose
    pose_landmarks[7] = {'x': 0.45, 'y': 0.18, 'z': 0.0}    # L ear
    pose_landmarks[8] = {'x': 0.55, 'y': 0.18, 'z': 0.0}    # R ear
    pose_landmarks[11] = {'x': 0.4, 'y': 0.4, 'z': 0.0}     # L shoulder
    pose_landmarks[12] = {'x': 0.6, 'y': 0.4, 'z': 0.0}     # R shoulder
    pose_landmarks[15] = {'x': 0.4, 'y': 0.1, 'z': 0.0}     # L wrist (above nose)
    pose_landmarks[16] = {'x': 0.6, 'y': 0.1, 'z': 0.0}     # R wrist (above nose)
    pose_landmarks[23] = {'x': 0.45, 'y': 0.8, 'z': 0.0}    # Hips
    pose_landmarks[24] = {'x': 0.55, 'y': 0.8, 'z': 0.0}

    pose_data = [{
        'landmarks': pose_landmarks,
        'world_landmarks': pose_landmarks
    }]

    # 2. Setup Mock Face (Smiling)
    face_landmarks_active = create_mock_landmarks(478)
    set_eye_landmarks(face_landmarks_active, 0.45, 0.55) # Stable gaze coordinates
    
    face_data = [{
        'landmarks': face_landmarks_active,
        'blendshapes': [
            {'category_name': 'mouthSmileLeft', 'score': 0.9},
            {'category_name': 'mouthSmileRight', 'score': 0.9},
            {'category_name': 'browDownLeft', 'score': 0.1},
            {'category_name': 'browDownRight', 'score': 0.1}
        ]
    }]

    # 3. Setup Mock Hand (Open Palm)
    hand_landmarks = create_mock_landmarks(21)
    hand_landmarks[0] = {'x': 0.5, 'y': 0.9, 'z': 0.0}      # Wrist
    hand_landmarks[9] = {'x': 0.5, 'y': 0.6, 'z': 0.0}      # Middle MCP
    
    # Extended fingers
    hand_landmarks[5] = {'x': 0.45, 'y': 0.6, 'z': 0.0}
    hand_landmarks[6] = {'x': 0.45, 'y': 0.5, 'z': 0.0}
    hand_landmarks[7] = {'x': 0.45, 'y': 0.4, 'z': 0.0}
    hand_landmarks[8] = {'x': 0.45, 'y': 0.3, 'z': 0.0}
    
    hand_landmarks[10] = {'x': 0.5, 'y': 0.5, 'z': 0.0}
    hand_landmarks[11] = {'x': 0.5, 'y': 0.4, 'z': 0.0}
    hand_landmarks[12] = {'x': 0.5, 'y': 0.28, 'z': 0.0}
    
    hand_landmarks[13] = {'x': 0.55, 'y': 0.6, 'z': 0.0}
    hand_landmarks[14] = {'x': 0.55, 'y': 0.52, 'z': 0.0}
    hand_landmarks[15] = {'x': 0.55, 'y': 0.42, 'z': 0.0}
    hand_landmarks[16] = {'x': 0.55, 'y': 0.32, 'z': 0.0}
    
    hand_landmarks[17] = {'x': 0.6, 'y': 0.6, 'z': 0.0}
    hand_landmarks[18] = {'x': 0.6, 'y': 0.55, 'z': 0.0}
    hand_landmarks[19] = {'x': 0.6, 'y': 0.48, 'z': 0.0}
    hand_landmarks[20] = {'x': 0.6, 'y': 0.4, 'z': 0.0}
    
    # Thumb spread
    hand_landmarks[1] = {'x': 0.47, 'y': 0.8, 'z': 0.0}
    hand_landmarks[2] = {'x': 0.42, 'y': 0.75, 'z': 0.0}
    hand_landmarks[3] = {'x': 0.37, 'y': 0.7, 'z': 0.0}
    hand_landmarks[4] = {'x': 0.32, 'y': 0.65, 'z': 0.0}

    hands_data = [{
        'handedness': 'Right',
        'landmarks': hand_landmarks
    }]

    # ----------------------------------------------------
    # PHASE A: Calibration (0 to 10000 ms)
    # ----------------------------------------------------
    neutral_pose = [{
        'landmarks': create_mock_landmarks(33),
        'world_landmarks': create_mock_landmarks(33)
    }]
    neutral_pose[0]['landmarks'][11] = {'x': 0.4, 'y': 0.4, 'z': 0.0}
    neutral_pose[0]['landmarks'][12] = {'x': 0.6, 'y': 0.4, 'z': 0.0}
    neutral_pose[0]['landmarks'][7] = {'x': 0.45, 'y': 0.18, 'z': 0.0}
    neutral_pose[0]['landmarks'][8] = {'x': 0.55, 'y': 0.18, 'z': 0.0}

    neutral_face = [{
        'landmarks': create_mock_landmarks(478),
        'blendshapes': [
            {'category_name': 'mouthSmileLeft', 'score': 0.05},
            {'category_name': 'mouthSmileRight', 'score': 0.05},
            {'category_name': 'browDownLeft', 'score': 0.05},
            {'category_name': 'browDownRight', 'score': 0.05},
            {'category_name': 'eyeSquintLeft', 'score': 0.05},
            {'category_name': 'eyeSquintRight', 'score': 0.05},
            {'category_name': 'noseWrinkleLeft', 'score': 0.05},
            {'category_name': 'noseWrinkleRight', 'score': 0.05},
            {'category_name': 'jawOpen', 'score': 0.05},
            {'category_name': 'eyeWideLeft', 'score': 0.05},
            {'category_name': 'eyeWideRight', 'score': 0.05},
            {'category_name': 'browInnerUp', 'score': 0.05},
            {'category_name': 'browOuterUpLeft', 'score': 0.05},
            {'category_name': 'browOuterUpRight', 'score': 0.05},
            {'category_name': 'tongueOut', 'score': 0.05},
            {'category_name': 'mouthPucker', 'score': 0.05}
        ]
    }]
    set_eye_landmarks(neutral_face[0]['landmarks'], 0.45, 0.55)

    print("Running 10s calibration simulation...")
    for t_ms in range(0, 10000, 1000):
        evaluator.evaluate(hands_data=[], pose_data=neutral_pose, face_data=neutral_face, timestamp_ms=t_ms)

    # ----------------------------------------------------
    # PHASE B: Stable Gaze Test (10100 to 11000 ms)
    # Feed 10 frames of perfectly centered pupil positions
    # ----------------------------------------------------
    print("Testing Gaze Stability (Stable Gaze)...")
    stable_face = [{
        'landmarks': create_mock_landmarks(478),
        'blendshapes': [
            {'category_name': 'mouthSmileLeft', 'score': 0.9},
            {'category_name': 'mouthSmileRight', 'score': 0.9}
        ]
    }]
    set_eye_landmarks(stable_face[0]['landmarks'], 0.45, 0.55)

    for t_ms in range(10100, 11000, 100):
        evaluator.evaluate(hands_data=[], pose_data=neutral_pose, face_data=stable_face, timestamp_ms=t_ms)

    # Evaluate final stable frame
    results = evaluator.evaluate(hands_data, pose_data, face_data, timestamp_ms=11000)
    
    print("\nStable Phase Results:")
    print(f"Hand Gestures (Right): {results['hands']['Right']}")
    print(f"Pose Gestures: {results['pose']}")
    print(f"Body Language: {results['body_language']}")
    print(f"Blink Count: {results.get('blink_count', 0)}")

    # Assert stable gaze and hand facing
    assert "Open Palm" in results['hands']['Right']
    assert "Palm Facing" in results['hands']['Right'], "Expected Right hand to be Palm Facing"
    assert "Hands Up" in results['pose']
    assert any("Smiling" in bl for bl in results['body_language'])
    assert "Stable Gaze (Focused/Calm)" in results['body_language'], "Expected Stable Gaze to be detected"

    # ----------------------------------------------------
    # PHASE B.2: Additional Micro-expressions Test (11010 to 11080 ms)
    # ----------------------------------------------------
    print("\nTesting Additional Micro-expressions (Squinting, Mouth Open, Smirks, Eyebrow Raises, Tongue Out)...")
    
    # 1. Squinting Test
    squint_face = [{
        'landmarks': create_mock_landmarks(478),
        'blendshapes': [
            {'category_name': 'eyeSquintLeft', 'score': 0.8},
            {'category_name': 'eyeSquintRight', 'score': 0.8}
        ]
    }]
    set_eye_landmarks(squint_face[0]['landmarks'], 0.45, 0.55)
    res = evaluator.evaluate(hands_data=[], pose_data=neutral_pose, face_data=squint_face, timestamp_ms=11010)
    print(f"Squinting results: {res['body_language']}")
    assert "Squinting (Eyes Narrowed)" in res['body_language']
    
    # 2. Mouth Open Test
    mouth_open_face = [{
        'landmarks': create_mock_landmarks(478),
        'blendshapes': [
            {'category_name': 'jawOpen', 'score': 0.8},
            {'category_name': 'eyeWideLeft', 'score': 0.05},
            {'category_name': 'eyeWideRight', 'score': 0.05}
        ]
    }]
    set_eye_landmarks(mouth_open_face[0]['landmarks'], 0.45, 0.55)
    res = evaluator.evaluate(hands_data=[], pose_data=neutral_pose, face_data=mouth_open_face, timestamp_ms=11020)
    print(f"Mouth Open results: {res['body_language']}")
    assert "Mouth Open (Parted Lips)" in res['body_language']
    
    # 3. Smirking (Left Side) Test
    smirk_left_face = [{
        'landmarks': create_mock_landmarks(478),
        'blendshapes': [
            {'category_name': 'mouthSmileLeft', 'score': 0.8},
            {'category_name': 'mouthSmileRight', 'score': 0.05}
        ]
    }]
    set_eye_landmarks(smirk_left_face[0]['landmarks'], 0.45, 0.55)
    res = evaluator.evaluate(hands_data=[], pose_data=neutral_pose, face_data=smirk_left_face, timestamp_ms=11030)
    print(f"Smirking Left results: {res['body_language']}")
    assert "Smirking (Left Side)" in res['body_language']

    # 4. Smirking (Right Side) Test
    smirk_right_face = [{
        'landmarks': create_mock_landmarks(478),
        'blendshapes': [
            {'category_name': 'mouthSmileLeft', 'score': 0.05},
            {'category_name': 'mouthSmileRight', 'score': 0.8}
        ]
    }]
    set_eye_landmarks(smirk_right_face[0]['landmarks'], 0.45, 0.55)
    res = evaluator.evaluate(hands_data=[], pose_data=neutral_pose, face_data=smirk_right_face, timestamp_ms=11040)
    print(f"Smirking Right results: {res['body_language']}")
    assert "Smirking (Right Side)" in res['body_language']

    # 5. Eyebrows Raised Test
    eyebrows_raised_face = [{
        'landmarks': create_mock_landmarks(478),
        'blendshapes': [
            {'category_name': 'browOuterUpLeft', 'score': 0.8},
            {'category_name': 'browOuterUpRight', 'score': 0.8}
        ]
    }]
    set_eye_landmarks(eyebrows_raised_face[0]['landmarks'], 0.45, 0.55)
    res = evaluator.evaluate(hands_data=[], pose_data=neutral_pose, face_data=eyebrows_raised_face, timestamp_ms=11050)
    print(f"Eyebrows Raised results: {res['body_language']}")
    assert "Eyebrows Raised (Surprise/Questioning)" in res['body_language']

    # 6. Skeptical Eyebrow Raise (Left) Test
    skeptical_left_face = [{
        'landmarks': create_mock_landmarks(478),
        'blendshapes': [
            {'category_name': 'browOuterUpLeft', 'score': 0.8},
            {'category_name': 'browOuterUpRight', 'score': 0.05}
        ]
    }]
    set_eye_landmarks(skeptical_left_face[0]['landmarks'], 0.45, 0.55)
    res = evaluator.evaluate(hands_data=[], pose_data=neutral_pose, face_data=skeptical_left_face, timestamp_ms=11060)
    print(f"Skeptical Left results: {res['body_language']}")
    assert "Skeptical Eyebrow Raise (Left)" in res['body_language']

    # 7. Skeptical Eyebrow Raise (Right) Test
    skeptical_right_face = [{
        'landmarks': create_mock_landmarks(478),
        'blendshapes': [
            {'category_name': 'browOuterUpLeft', 'score': 0.05},
            {'category_name': 'browOuterUpRight', 'score': 0.8}
        ]
    }]
    set_eye_landmarks(skeptical_right_face[0]['landmarks'], 0.45, 0.55)
    res = evaluator.evaluate(hands_data=[], pose_data=neutral_pose, face_data=skeptical_right_face, timestamp_ms=11070)
    print(f"Skeptical Right results: {res['body_language']}")
    assert "Skeptical Eyebrow Raise (Right)" in res['body_language']

    # 8. Tongue Out Test
    tongue_out_face = [{
        'landmarks': create_mock_landmarks(478),
        'blendshapes': [
            {'category_name': 'tongueOut', 'score': 0.8}
        ]
    }]
    set_eye_landmarks(tongue_out_face[0]['landmarks'], 0.45, 0.55)
    res = evaluator.evaluate(hands_data=[], pose_data=neutral_pose, face_data=tongue_out_face, timestamp_ms=11080)
    print(f"Tongue Out results: {res['body_language']}")
    assert "Tongue Out (Playful/Teasing)" in res['body_language']

    # ----------------------------------------------------
    # PHASE C: Blink Test & Rapid Blinking Test (11100 to 12400 ms)
    # ----------------------------------------------------
    print("\nTesting Blink Detection & Rapid Blinking...")
    # Close eyes
    closed_face = [{
        'landmarks': create_mock_landmarks(478),
        'blendshapes': [
            {'category_name': 'eyeBlinkLeft', 'score': 0.95},
            {'category_name': 'eyeBlinkRight', 'score': 0.95}
        ]
    }]
    # Open eyes
    open_face = [{
        'landmarks': create_mock_landmarks(478),
        'blendshapes': [
            {'category_name': 'eyeBlinkLeft', 'score': 0.05},
            {'category_name': 'eyeBlinkRight', 'score': 0.05}
        ]
    }]
    set_eye_landmarks(open_face[0]['landmarks'], 0.45, 0.55)

    # Cycle 1
    evaluator.evaluate(hands_data=[], pose_data=neutral_pose, face_data=closed_face, timestamp_ms=11500)
    blink_results = evaluator.evaluate(hands_data=[], pose_data=neutral_pose, face_data=open_face, timestamp_ms=12000)
    assert blink_results.get('blink_count', 0) == 1

    # Cycle 2
    evaluator.evaluate(hands_data=[], pose_data=neutral_pose, face_data=closed_face, timestamp_ms=12100)
    evaluator.evaluate(hands_data=[], pose_data=neutral_pose, face_data=open_face, timestamp_ms=12200)

    # Cycle 3
    evaluator.evaluate(hands_data=[], pose_data=neutral_pose, face_data=closed_face, timestamp_ms=12300)
    blink_results = evaluator.evaluate(hands_data=[], pose_data=neutral_pose, face_data=open_face, timestamp_ms=12400)
    
    print(f"Blink Count after 3 cycles: {blink_results.get('blink_count', 0)}")
    print(f"Body language after 3 rapid blinks: {blink_results.get('body_language')}")
    assert blink_results.get('blink_count', 0) == 3
    assert "Rapid/Irregular Blinking (Tension)" in blink_results.get('body_language'), "Expected Rapid Blinking to be detected"

    # ----------------------------------------------------
    # PHASE D: Shifty Gaze & REM Test (12500 to 13900 ms)
    # Feed 15 frames of alternating pupil coordinates to trigger saccades
    # ----------------------------------------------------
    print("\nTesting Gaze Stability (Shifty Gaze & REMs)...")
    for idx, t_ms in enumerate(range(12500, 13900, 100)):
        shifty_face = [{
            'landmarks': create_mock_landmarks(478),
            'blendshapes': []
        }]
        offset = -0.015 if idx % 2 == 0 else 0.015
        set_eye_landmarks(shifty_face[0]['landmarks'], 0.45 + offset, 0.55 + offset)
        shifty_results = evaluator.evaluate(hands_data=[], pose_data=neutral_pose, face_data=shifty_face, timestamp_ms=t_ms)

    print(f"Body language after shifty phase: {shifty_results.get('body_language')}")
    assert "Shifty Gaze (Restless/Anxious)" in shifty_results.get('body_language'), "Expected Shifty Gaze to be detected"
    assert "Rapid Eye Movements (Restless)" in shifty_results.get('body_language'), "Expected REMs to be detected"

    # ----------------------------------------------------
    # PHASE E: Dynamic Waving Test (13600 to 15500 ms)
    # Feed 15 frames where Right Wrist Y is raised and X oscillates
    # ----------------------------------------------------
    print("\nTesting Active Waving (Right Hand Waving)...")
    for idx, t_ms in enumerate(range(13600, 15500, 100)):
        wave_pose = [{
            'landmarks': create_mock_landmarks(33),
            'world_landmarks': create_mock_landmarks(33)
        }]
        wave_pose[0]['landmarks'][11] = {'x': 0.4, 'y': 0.4, 'z': 0.0} # L shoulder
        wave_pose[0]['landmarks'][12] = {'x': 0.6, 'y': 0.4, 'z': 0.0} # R shoulder
        wave_pose[0]['landmarks'][13] = {'x': 0.35, 'y': 0.6, 'z': 0.0} # L elbow
        wave_pose[0]['landmarks'][14] = {'x': 0.65, 'y': 0.6, 'z': 0.0} # R elbow
        wave_pose[0]['landmarks'][15] = {'x': 0.35, 'y': 0.8, 'z': 0.0} # L wrist (down)
        
        # Right wrist is raised (y=0.2 < shoulder y=0.4) and swings horizontally
        x_offset = -0.04 if idx % 2 == 0 else 0.04
        wave_pose[0]['landmarks'][16] = {'x': 0.65 + x_offset, 'y': 0.2, 'z': 0.0}
        
        wave_face = [{
            'landmarks': create_mock_landmarks(478),
            'blendshapes': []
        }]
        set_eye_landmarks(wave_face[0]['landmarks'], 0.45, 0.55)
        
        wave_results = evaluator.evaluate(hands_data=[], pose_data=wave_pose, face_data=wave_face, timestamp_ms=t_ms)
        
    print(f"Body language after waving phase: {wave_results.get('body_language')}")
    assert "Right Hand Waving (Active)" in wave_results.get('body_language'), "Expected Right Hand Waving (Active) to be detected"

    print("\nAll mock test cases passed successfully!")

if __name__ == "__main__":
    try:
        test_full_gesture_system()
    except AssertionError as e:
        print(f"\nAssertion Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)
