from gesture_recognition.utils import dist3d, dist2d, get_angle

def recognize_pose_gestures(pose_data):
    """
    Recognize 30 pose gestures from pose landmark data.
    pose_data: list of dicts returned by PoseDetectorWrapper.get_latest_data()
    
    Returns: list of detected gesture names.
    """
    gestures = []
    if not pose_data or len(pose_data) == 0:
        return gestures

    landmarks = pose_data[0]['landmarks']
    
    # Extract landmarks for easy reference
    nose = landmarks[0]
    l_ear = landmarks[7]
    r_ear = landmarks[8]
    l_shoulder = landmarks[11]
    r_shoulder = landmarks[12]
    l_elbow = landmarks[13]
    r_elbow = landmarks[14]
    l_wrist = landmarks[15]
    r_wrist = landmarks[16]
    l_hip = landmarks[23]
    r_hip = landmarks[24]
    l_knee = landmarks[25]
    r_knee = landmarks[26]
    l_ankle = landmarks[27]
    r_ankle = landmarks[28]
    l_heel = landmarks[29]
    r_heel = landmarks[30]
    l_foot = landmarks[31]
    r_foot = landmarks[32]

    # Reference scales
    shoulder_width = dist3d(l_shoulder, r_shoulder)
    if shoulder_width == 0:
        shoulder_width = 0.001
        
    hip_width = dist3d(l_hip, r_hip)
    torso_len = dist3d(
        {'x': (l_shoulder['x'] + r_shoulder['x'])/2, 'y': (l_shoulder['y'] + r_shoulder['y'])/2, 'z': (l_shoulder['z'] + r_shoulder['z'])/2},
        {'x': (l_hip['x'] + r_hip['x'])/2, 'y': (l_hip['y'] + r_hip['y'])/2, 'z': (l_hip['z'] + r_hip['z'])/2}
    )
    if torso_len == 0:
        torso_len = 0.001

    # Y-coordinates are inverted (lower Y means higher up on screen)

    # 1. Hands Up (Surrender / Victory)
    # Both wrists above nose
    if l_wrist['y'] < nose['y'] and r_wrist['y'] < nose['y']:
        gestures.append("Hands Up")

    # 2. T-Pose
    # Both elbows and wrists level with shoulders, arms extended horizontally
    l_arm_straight = get_angle(l_shoulder, l_elbow, l_wrist) > 150
    r_arm_straight = get_angle(r_shoulder, r_elbow, r_wrist) > 150
    l_shoulder_aligned = abs(l_wrist['y'] - l_shoulder['y']) < 0.2 * torso_len
    r_shoulder_aligned = abs(r_wrist['y'] - r_shoulder['y']) < 0.2 * torso_len
    if l_arm_straight and r_arm_straight and l_shoulder_aligned and r_shoulder_aligned:
        gestures.append("T-Pose")

    # 3. Right Hand Wave
    # Right wrist above right shoulder and near head
    if r_wrist['y'] < r_shoulder['y'] and dist3d(r_wrist, r_ear) < 0.8 * shoulder_width:
        gestures.append("Right Hand Wave")

    # 4. Left Hand Wave
    # Left wrist above left shoulder and near head
    if l_wrist['y'] < l_shoulder['y'] and dist3d(l_wrist, l_ear) < 0.8 * shoulder_width:
        gestures.append("Left Hand Wave")

    # 5. Arms Crossed
    # Left wrist close to right elbow and right wrist close to left elbow
    if dist3d(l_wrist, r_elbow) < 0.7 * shoulder_width and dist3d(r_wrist, l_elbow) < 0.7 * shoulder_width:
        if l_wrist['y'] > l_shoulder['y'] and r_wrist['y'] > r_shoulder['y']:
            gestures.append("Arms Crossed")

    # 6. Hands on Hips
    # Both wrists near hips
    if dist3d(l_wrist, l_hip) < 0.6 * shoulder_width and dist3d(r_wrist, r_hip) < 0.6 * shoulder_width:
        gestures.append("Hands on Hips")

    # 7. Salute
    # Right hand near forehead/right eye
    if r_wrist['y'] < nose['y'] and dist3d(r_wrist, r_ear) < 0.5 * shoulder_width:
        # Check that right elbow is bent
        if get_angle(r_shoulder, r_elbow, r_wrist) < 120:
            gestures.append("Salute")

    # 8. Right Arm Pointing Right
    # Right arm extended straight to the right (r_wrist x-coordinate is significantly smaller than r_shoulder)
    # Note: Webcam might be flipped, but in camera coordinates, right arm points right (x smaller or larger depending on flip)
    # Let's check: right arm straight, and wrist is level with shoulder, and pointing away from body
    if r_arm_straight and abs(r_wrist['y'] - r_shoulder['y']) < 0.3 * torso_len:
        # Distance from wrist to left shoulder is greater than elbow to left shoulder
        if dist3d(r_wrist, l_shoulder) > dist3d(r_elbow, l_shoulder):
            gestures.append("Right Arm Pointing Right")

    # 9. Left Arm Pointing Left
    # Left arm extended straight to the left
    if l_arm_straight and abs(l_wrist['y'] - l_shoulder['y']) < 0.3 * torso_len:
        if dist3d(l_wrist, r_shoulder) > dist3d(l_elbow, r_shoulder):
            gestures.append("Left Arm Pointing Left")

    # 10. Right Arm Pointing Up
    # Right arm straight and right wrist high above head
    if r_arm_straight and r_wrist['y'] < nose['y'] - 0.3 * torso_len:
        gestures.append("Right Arm Pointing Up")

    # 11. Left Arm Pointing Up
    # Left arm straight and left wrist high above head
    if l_arm_straight and l_wrist['y'] < nose['y'] - 0.3 * torso_len:
        gestures.append("Left Arm Pointing Up")

    # 12. Right Hand on Heart
    # Right wrist close to chest center / left shoulder
    if dist3d(r_wrist, l_shoulder) < 0.6 * shoulder_width and r_wrist['y'] > r_shoulder['y']:
        if dist3d(r_wrist, r_shoulder) > 0.4 * shoulder_width:
            gestures.append("Right Hand on Heart")

    # 13. Left Hand on Heart
    # Left wrist close to chest center / right shoulder
    if dist3d(l_wrist, r_shoulder) < 0.6 * shoulder_width and l_wrist['y'] > l_shoulder['y']:
        if dist3d(l_wrist, l_shoulder) > 0.4 * shoulder_width:
            gestures.append("Left Hand on Heart")

    # 14. Flexing Biceps (Double)
    # Both wrists near head, elbows bent, elbows level with shoulders
    if r_wrist['y'] < r_shoulder['y'] and l_wrist['y'] < l_shoulder['y']:
        if get_angle(r_shoulder, r_elbow, r_wrist) < 110 and get_angle(l_shoulder, l_elbow, l_wrist) < 110:
            if abs(r_elbow['y'] - r_shoulder['y']) < 0.3 * torso_len and abs(l_elbow['y'] - l_shoulder['y']) < 0.3 * torso_len:
                gestures.append("Flexing Biceps")

    # 15. Thinking Pose (Hand to Chin)
    # Either hand close to nose/chin
    if dist3d(r_wrist, nose) < 0.6 * shoulder_width or dist3d(l_wrist, nose) < 0.6 * shoulder_width:
        if r_wrist['y'] > nose['y'] or l_wrist['y'] > nose['y']:
            gestures.append("Thinking Pose")

    # 16. Shielding Eyes
    # One hand directly above eyes/forehead
    if (dist3d(r_wrist, nose) < 0.6 * shoulder_width and r_wrist['y'] < nose['y'] and abs(r_wrist['x'] - nose['x']) < 0.2 * shoulder_width) or \
       (dist3d(l_wrist, nose) < 0.6 * shoulder_width and l_wrist['y'] < nose['y'] and abs(l_wrist['x'] - nose['x']) < 0.2 * shoulder_width):
        gestures.append("Shielding Eyes")

    # 17. Dab
    # One arm bent across face, other extended straight up/out
    l_bent = get_angle(l_shoulder, l_elbow, l_wrist) < 90
    r_bent = get_angle(r_shoulder, r_elbow, r_wrist) < 90
    if (r_bent and l_arm_straight and dist3d(r_wrist, nose) < 0.5 * shoulder_width) or \
       (l_bent and r_arm_straight and dist3d(l_wrist, nose) < 0.5 * shoulder_width):
        gestures.append("Dab")

    # 18. Bow / Leaning Forward
    # Nose is lower than shoulders, or shoulders are low relative to hips compared to standard pose
    # We check if torso length is compressed vertically (nose/shoulders close to hips in Y, but z increases)
    # Let's say: nose Y is very close to hips relative to typical torso_len
    if torso_len < 0.7 * shoulder_width:
        gestures.append("Bow / Leaning Forward")

    # 19. Tree Pose (Yoga)
    # One ankle/foot resting on or near the opposite knee
    if dist3d(r_ankle, l_knee) < 0.5 * hip_width:
        gestures.append("Tree Pose (Right Leg)")
        gestures.append("Tree Pose")
    elif dist3d(l_ankle, r_knee) < 0.5 * hip_width:
        gestures.append("Tree Pose (Left Leg)")
        gestures.append("Tree Pose")

    # 20. Warrior Pose (Yoga)
    # Arms extended horizontally (T-pose) and legs wide apart
    if "T-Pose" in gestures and dist3d(l_ankle, r_ankle) > 1.8 * shoulder_width:
        gestures.append("Warrior Pose")

    # 21. Squatting
    # Hips low, knee angle bent (< 120 deg)
    l_knee_ang = get_angle(l_hip, l_knee, l_ankle)
    r_knee_ang = get_angle(r_hip, r_knee, r_ankle)
    if l_knee_ang < 120 and r_knee_ang < 120 and l_hip['y'] > (l_shoulder['y'] + l_ankle['y'])/2:
        gestures.append("Squatting")

    # 22. Jumping Jack (Mid-air)
    # Wrists above shoulders, arms wide, feet wide
    if l_wrist['y'] < l_shoulder['y'] and r_wrist['y'] < r_shoulder['y']:
        if dist3d(l_wrist, r_wrist) > 1.5 * shoulder_width and dist3d(l_ankle, r_ankle) > 1.5 * shoulder_width:
            gestures.append("Jumping Jack")

    # 23. Right Leg Raised
    # Right knee bent, right ankle higher than left ankle
    if r_knee_ang < 110 and r_ankle['y'] < l_ankle['y'] - 0.15 * torso_len:
        gestures.append("Right Leg Raised")

    # 24. Left Leg Raised
    # Left knee bent, left ankle higher than right ankle
    if l_knee_ang < 110 and l_ankle['y'] < r_ankle['y'] - 0.15 * torso_len:
        gestures.append("Left Leg Raised")

    # 25. Hands Clapping
    # Both wrists very close to each other, near chest
    if dist3d(l_wrist, r_wrist) < 0.3 * shoulder_width and l_wrist['y'] > l_shoulder['y'] and l_wrist['y'] < l_hip['y']:
        gestures.append("Hands Clapping")

    # 26. Right Hand on Head
    # Right hand close to top of head / ear
    if dist3d(r_wrist, r_ear) < 0.4 * shoulder_width and r_wrist['y'] < r_ear['y']:
        gestures.append("Right Hand on Head")

    # 27. Left Hand on Head
    # Left hand close to top of head / ear
    if dist3d(l_wrist, l_ear) < 0.4 * shoulder_width and l_wrist['y'] < l_ear['y']:
        gestures.append("Left Hand on Head")

    # 28. Hugging Pose
    # Arms open wide, elbows level with shoulders, but not straight (slightly bent forward/inwards)
    if 90 < get_angle(l_shoulder, l_elbow, l_wrist) < 140 and 90 < get_angle(r_shoulder, r_elbow, r_wrist) < 140:
        if dist3d(l_wrist, r_wrist) > 1.2 * shoulder_width:
            gestures.append("Hugging Pose")

    # 29. Kick Right
    # Right ankle extended forward/up (right knee straight and ankle high)
    if r_knee_ang > 150 and r_ankle['y'] < l_knee['y']:
        gestures.append("Kick Right")

    # 30. Kick Left
    # Left ankle extended forward/up (left knee straight and ankle high)
    if l_knee_ang > 150 and l_ankle['y'] < r_knee['y']:
        gestures.append("Kick Left")

    return list(set(gestures))
