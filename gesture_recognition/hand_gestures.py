from gesture_recognition.utils import dist3d, dist2d, get_angle, get_finger_states

def detect_hand_facing(landmarks, handedness):
    """
    Detects whether the palm or the back of the hand is facing the camera.
    Uses the 2D cross product of vectors Wrist(0)->Index_MCP(5) and Wrist(0)->Pinky_MCP(17).
    """
    x0, y0 = landmarks[0]['x'], landmarks[0]['y']
    x5, y5 = landmarks[5]['x'], landmarks[5]['y']
    x17, y17 = landmarks[17]['x'], landmarks[17]['y']
    
    # Vectors in 2D
    v1_x, v1_y = x5 - x0, y5 - y0
    v2_x, v2_y = x17 - x0, y17 - y0
    
    # 2D cross product
    cross = v1_x * v2_y - v1_y * v2_x
    
    # Mirror/raw flip adjustment
    if handedness == 'Right':
        return 'Back' if cross > 0 else 'Palm'
    else:
        return 'Back' if cross < 0 else 'Palm'

def recognize_hand_gestures(hands_data, pose_data=None, face_data=None):
    """
    Recognize hand gestures from the hands_data, with optional pose and face coordinates.
    hands_data: list of dicts returned by HandDetectorWrapper.get_latest_data()
    pose_data: list of dicts returned by PoseDetectorWrapper.get_latest_data()
    face_data: list of dicts returned by FaceDetectorWrapper.get_latest_data()
    
    Returns: dict mapping handedness ('Left'/'Right') to a list of detected gesture names.
    Also returns a 'DoubleHand' key for gestures involving both hands.
    """
    results = {
        'Left': [],
        'Right': [],
        'DoubleHand': []
    }
    
    if not hands_data:
        return results

    # Pre-extract hands for easy access
    left_hand = None
    right_hand = None
    for hand in hands_data:
        if hand['handedness'] == 'Left':
            left_hand = hand
        elif hand['handedness'] == 'Right':
            right_hand = hand

    # Get face & pose references if available
    pose_landmarks = None
    if pose_data and len(pose_data) > 0:
        pose_landmarks = pose_data[0]['landmarks']
        
    face_landmarks = None
    if face_data and len(face_data) > 0:
        face_landmarks = face_data[0]['landmarks']

    # Process individual hands
    for hand in hands_data:
        handedness = hand['handedness']
        landmarks = hand['landmarks']
        
        # Since the frame coordinates are mirrored, the visual geometry has inverted chirality.
        # We compute geometry_handedness to represent the visual appearance of the hand in the image.
        geometry_handedness = 'Left' if handedness == 'Right' else 'Right'
        
        # Get finger extension states and ratios
        f_states = get_finger_states(landmarks)
        t = f_states['thumb']
        i = f_states['index']
        m = f_states['middle']
        r = f_states['ring']
        p = f_states['pinky']
        
        tr = f_states['thumb_ratio']
        ir = f_states['index_ratio']
        mr = f_states['middle_ratio']
        rr = f_states['ring_ratio']
        pr = f_states['pinky_ratio']
        
        hand_scale = f_states['hand_scale']
        thumb_to_index_mcp = f_states['thumb_to_index_mcp']
        
        # Common tip coordinates
        wrist = landmarks[0]
        t_tip = landmarks[4]
        i_tip = landmarks[8]
        m_tip = landmarks[12]
        r_tip = landmarks[16]
        p_tip = landmarks[20]
        
        # MCPs
        i_mcp = landmarks[5]
        m_mcp = landmarks[9]
        r_mcp = landmarks[13]
        p_mcp = landmarks[17]

        # ----------------------------------------------------
        # Standard/Common Hand Gestures (20 Gestures)
        # ----------------------------------------------------
        gestures = []
        
        # Detect facing (Palm vs Back of Hand)
        facing = detect_hand_facing(landmarks, geometry_handedness)
        if facing == 'Back':
            gestures.append("Back of Hand")
        else:
            gestures.append("Palm Facing")
        
        # 1. Open Palm
        if t and i and m and r and p:
            gestures.append("Open Palm")
            
        # 2. Fist
        if not t and not i and not m and not r and not p:
            gestures.append("Fist")

        # 3. Thumbs Up
        if t and not i and not m and not r and not p and t_tip['y'] < i_mcp['y']:
            gestures.append("Thumbs Up")
            
        # 4. Thumbs Down
        if t and not i and not m and not r and not p and t_tip['y'] > wrist['y']:
            gestures.append("Thumbs Down")
            
        # 5. Index Pointing Up
        if not t and i and not m and not r and not p and i_tip['y'] < i_mcp['y']:
            gestures.append("Index Pointing Up")

        # 6. Index Pointing Forward
        # Tip is extended, others curled, but hand is horizontal or pointing forward (z-depth)
        if not t and i and not m and not r and not p and abs(i_tip['y'] - i_mcp['y']) < 0.2:
            gestures.append("Index Pointing Forward")

        # 7. Peace Sign / 8. Victory
        if not t and i and m and not r and not p and dist3d(i_tip, m_tip) > 0.25 * hand_scale:
            gestures.append("Peace Sign")
            gestures.append("Victory")

        # 9. Rock On (Sign of Horns)
        if i and p and not m and not r:
            gestures.append("Rock On")

        # 10. OK Sign
        if m and r and p and dist3d(t_tip, i_tip) < 0.2 * hand_scale:
            gestures.append("OK Sign")

        # 11. Spock (Vulcan Salute)
        if t and i and m and r and p:
            d_i_m = dist3d(i_tip, m_tip)
            d_m_r = dist3d(m_tip, r_tip)
            d_r_p = dist3d(r_tip, p_tip)
            if d_m_r > 0.35 * hand_scale and d_i_m < 0.25 * hand_scale and d_r_p < 0.25 * hand_scale:
                gestures.append("Spock (Vulcan Salute)")

        # 12. Spider-Man
        if t and i and p and not m and not r:
            gestures.append("Spider-Man")

        # 13. Pistol
        if t and i and not m and not r and not p and dist3d(t_tip, i_mcp) > 0.45 * hand_scale:
            gestures.append("Pistol")

        # 14. Three (European)
        if t and i and m and not r and not p:
            gestures.append("Three (European)")

        # 15. Three (American)
        if not t and i and m and r and not p:
            gestures.append("Three (American)")

        # 16. Four
        if not t and i and m and r and p:
            gestures.append("Four")

        # 17. Pinky Up
        if not t and not i and not m and not r and p:
            gestures.append("Pinky Up")

        # 18. Stop / Halt
        if t and i and m and r and p and i_tip['y'] < i_mcp['y']:
            # Hand orientation check: fingers pointing up, palm facing camera (typically z of tips < z of wrist)
            gestures.append("Stop / Halt")

        # 19. Claw
        if (0.45 < ir < 0.75) and (0.45 < mr < 0.75) and (0.45 < rr < 0.75) and (0.45 < pr < 0.75):
            gestures.append("Claw")

        # 20. Heart Hand (Single)
        # Thumb and index tips are close and curved, middle/ring/pinky curled
        if not m and not r and not p and dist3d(t_tip, i_tip) < 0.35 * hand_scale:
            # check that index is somewhat curled (curved index forming half-heart)
            if 0.4 < ir < 0.78 and 0.4 < tr < 0.85:
                gestures.append("Heart Hand (Single)")

        # 21. Call Me
        if t and p and not i and not m and not r:
            gestures.append("Call Me")

        # ----------------------------------------------------
        # ASL Fingerspelling Letters (24 Gestures)
        # ----------------------------------------------------
        
        # 22. ASL Letter A [Sign Language]
        # Fist with thumb resting on the side of index finger
        if not i and not m and not r and not p and not t and thumb_to_index_mcp < 0.48:
            # thumb tip should be near index MCP/PIP
            if dist3d(t_tip, landmarks[6]) < 0.35 * hand_scale:
                gestures.append("ASL Letter A [Sign Language]")

        # 23. ASL Letter B [Sign Language]
        # Open flat hand, thumb crossed in front of palm
        if i and m and r and p and not t:
            # Thumb tip is tucked inwards across palm (close to pinky MCP or middle MCP)
            if dist3d(t_tip, m_mcp) < 0.4 * hand_scale or dist3d(t_tip, r_mcp) < 0.4 * hand_scale:
                gestures.append("ASL Letter B [Sign Language]")

        # 24. ASL Letter C [Sign Language]
        # Curved hand forming 'C'. All fingers partially curled.
        if (0.45 < ir < 0.78) and (0.45 < mr < 0.78) and (0.45 < rr < 0.78) and (0.45 < pr < 0.78) and (0.45 < tr < 0.85):
            # Form C shape check: index tip to thumb tip is open, but both are curved
            if dist3d(t_tip, i_tip) > 0.4 * hand_scale:
                gestures.append("ASL Letter C [Sign Language]")

        # 25. ASL Letter D [Sign Language]
        # Index extended, middle, ring, pinky tips touch thumb tip
        if i and not m and not r and not p:
            if dist3d(t_tip, m_tip) < 0.3 * hand_scale and dist3d(t_tip, r_tip) < 0.3 * hand_scale:
                gestures.append("ASL Letter D [Sign Language]")

        # 26. ASL Letter E [Sign Language]
        # All fingers curled, but not a fist. Tips are tucked close to MCPs, thumb curled in front.
        if not t and not i and not m and not r and not p:
            # specific E check: fingers tips are curled but not overlapping, sitting on thumb
            if ir < 0.5 and mr < 0.5 and rr < 0.5 and pr < 0.5:
                if dist3d(t_tip, landmarks[8]) < 0.35 * hand_scale:
                    gestures.append("ASL Letter E [Sign Language]")

        # 27. ASL Letter F [Sign Language]
        # Index and thumb touch, middle, ring, pinky extended
        if not i and m and r and p and dist3d(t_tip, i_tip) < 0.25 * hand_scale:
            gestures.append("ASL Letter F [Sign Language]")

        # 28. ASL Letter G [Sign Language]
        # Index and thumb pointing horizontally (pointing left/right), others curled
        if t and i and not m and not r and not p:
            # G is horizontal. Index vector 5->8 is horizontal (y difference is small, x is large)
            idx_vector_y = abs(i_tip['y'] - i_mcp['y'])
            idx_vector_x = abs(i_tip['x'] - i_mcp['x'])
            if idx_vector_x > idx_vector_y:
                gestures.append("ASL Letter G [Sign Language]")

        # 29. ASL Letter H [Sign Language]
        # Index and middle extended horizontally, thumb and others curled
        if not t and i and m and not r and not p:
            idx_vector_y = abs(i_tip['y'] - i_mcp['y'])
            idx_vector_x = abs(i_tip['x'] - i_mcp['x'])
            if idx_vector_x > idx_vector_y:
                gestures.append("ASL Letter H [Sign Language]")

        # 30. ASL Letter I [Sign Language]
        # Pinky extended, others curled
        if not t and not i and not m and not r and p:
            gestures.append("ASL Letter I [Sign Language]")

        # 31. ASL Letter K [Sign Language]
        # Index and middle up, thumb pointing up touching middle PIP/MCP
        if i and m and not r and not p:
            # thumb tip close to index or middle MCP/PIP
            if dist3d(t_tip, landmarks[6]) < 0.3 * hand_scale or dist3d(t_tip, landmarks[10]) < 0.3 * hand_scale:
                # thumb tip is higher up
                if t_tip['y'] < wrist['y']:
                    gestures.append("ASL Letter K [Sign Language]")

        # 32. ASL Letter L [Sign Language]
        # Index and thumb extended forming L shape
        if t and i and not m and not r and not p:
            # angle index to thumb
            ang = get_angle(i_tip, wrist, t_tip)
            if 40 < ang < 110:
                # Index is vertical, thumb is horizontal
                if i_tip['y'] < i_mcp['y'] and abs(t_tip['y'] - i_mcp['y']) < 0.25 * hand_scale:
                    gestures.append("ASL Letter L [Sign Language]")

        # 33. ASL Letter M [Sign Language]
        # Fist, thumb tucked under index, middle, ring (near pinky MCP)
        if not t and not i and not m and not r and not p:
            if dist3d(t_tip, p_mcp) < 0.35 * hand_scale:
                gestures.append("ASL Letter M [Sign Language]")

        # 34. ASL Letter N [Sign Language]
        # Fist, thumb tucked under index and middle (near ring MCP)
        if not t and not i and not m and not r and not p:
            if dist3d(t_tip, r_mcp) < 0.35 * hand_scale and dist3d(t_tip, p_mcp) > 0.35 * hand_scale:
                gestures.append("ASL Letter N [Sign Language]")

        # 35. ASL Letter O [Sign Language]
        # All finger tips touching thumb tip to make a circle
        if not i and not m and not r and not p:
            if dist3d(t_tip, i_tip) < 0.28 * hand_scale and dist3d(t_tip, m_tip) < 0.28 * hand_scale and dist3d(t_tip, r_tip) < 0.28 * hand_scale:
                gestures.append("ASL Letter O [Sign Language]")

        # 36. ASL Letter P [Sign Language]
        # K shape pointing downwards
        if i and m and not r and not p:
            if i_tip['y'] > i_mcp['y']:  # pointing down
                gestures.append("ASL Letter P [Sign Language]")

        # 37. ASL Letter Q [Sign Language]
        # G shape pointing downwards
        if t and i and not m and not r and not p:
            if i_tip['y'] > i_mcp['y']:  # pointing down
                gestures.append("ASL Letter Q [Sign Language]")

        # 38. ASL Letter R [Sign Language]
        # Index and middle extended, crossed (tips are close, cross-plane)
        if i and m and not r and not p:
            if dist3d(i_tip, m_tip) < 0.15 * hand_scale:
                # check crossing (using geometry_handedness due to mirrored coordinates)
                if (geometry_handedness == 'Right' and i_tip['x'] > m_tip['x']) or (geometry_handedness == 'Left' and i_tip['x'] < m_tip['x']):
                    gestures.append("ASL Letter R [Sign Language]")

        # 39. ASL Letter S [Sign Language]
        # Fist, thumb in front of fingers
        if not t and not i and not m and not r and not p:
            # Thumb tip sits on index/middle fingers (y-coord close to PIPs)
            if dist3d(t_tip, landmarks[6]) < 0.3 * hand_scale or dist3d(t_tip, landmarks[10]) < 0.3 * hand_scale:
                # distinguish from A, M, N: S is right in front (check using geometry_handedness due to mirroring)
                if t_tip['x'] > i_mcp['x'] if geometry_handedness == 'Left' else t_tip['x'] < i_mcp['x']:
                    gestures.append("ASL Letter S [Sign Language]")

        # 40. ASL Letter T [Sign Language]
        # Fist, thumb tucked under index finger (between index and middle)
        if not t and not i and not m and not r and not p:
            if dist3d(t_tip, landmarks[6]) < 0.25 * hand_scale and dist3d(t_tip, m_mcp) < 0.3 * hand_scale:
                gestures.append("ASL Letter T [Sign Language]")

        # 41. ASL Letter U [Sign Language]
        # Index & middle together pointing up
        if i and m and not r and not p:
            if dist3d(i_tip, m_tip) < 0.15 * hand_scale:
                gestures.append("ASL Letter U [Sign Language]")

        # 42. ASL Letter V [Sign Language]
        # Index & middle spread (Peace shape)
        if i and m and not r and not p:
            if dist3d(i_tip, m_tip) >= 0.25 * hand_scale:
                gestures.append("ASL Letter V [Sign Language]")

        # 43. ASL Letter W [Sign Language]
        # Index, middle, ring extended & spread, thumb and pinky touch
        if i and m and r and not p:
            if dist3d(t_tip, p_tip) < 0.35 * hand_scale:
                gestures.append("ASL Letter W [Sign Language]")

        # 44. ASL Letter X [Sign Language]
        # Index finger hooked, others curled
        if not t and not m and not r and not p:
            if 0.35 < ir < 0.72:
                gestures.append("ASL Letter X [Sign Language]")

        # 45. ASL Letter Y [Sign Language]
        # Thumb and pinky extended, others curled
        if t and p and not i and not m and not r:
            gestures.append("ASL Letter Y [Sign Language]")

        # ----------------------------------------------------
        # Contextual ASL Words/Signs (Face/Body Relative)
        # ----------------------------------------------------
        
        # 46. ASL Sign 'Hello' [Sign Language]
        # Flat hand (Open Palm) near forehead
        if (t and i and m and r and p) or gestures.__contains__("Open Palm"):
            # Check proximity of hand to forehead/head in pose or face
            if face_landmarks:
                # Face landmarks: forehead is around 10
                forehead = face_landmarks[10]
                if dist3d(wrist, forehead) < 0.25:
                    gestures.append("ASL Sign 'Hello' [Sign Language]")
            elif pose_landmarks:
                nose = pose_landmarks[0]
                if dist3d(wrist, nose) < 0.35 and wrist['y'] < nose['y']:
                    gestures.append("ASL Sign 'Hello' [Sign Language]")

        # 47. ASL Sign 'Thank You' [Sign Language]
        # Flat hand starts near chin/mouth and moves forward
        if (t and i and m and r and p) or gestures.__contains__("Open Palm"):
            if face_landmarks:
                chin = face_landmarks[152]
                mouth = face_landmarks[13]
                if dist3d(wrist, mouth) < 0.22 or dist3d(t_tip, mouth) < 0.18:
                    gestures.append("ASL Sign 'Thank You' [Sign Language]")
            elif pose_landmarks:
                nose = pose_landmarks[0]
                if dist3d(wrist, nose) < 0.28:
                    gestures.append("ASL Sign 'Thank You' [Sign Language]")

        # 48. ASL Sign 'Yes' [Sign Language]
        # Fist tilting/nodding. We can annotate a Fist in vertical position as ASL Yes.
        if "Fist" in gestures:
            gestures.append("ASL Sign 'Yes' [Sign Language]")

        # 49. ASL Sign 'No' [Sign Language]
        # Index and middle snap to touch thumb
        if not r and not p:
            if dist3d(t_tip, i_tip) < 0.22 * hand_scale and dist3d(t_tip, m_tip) < 0.22 * hand_scale:
                # index and middle curled ratios are low (meaning closed/snapping)
                if ir < 0.65 and mr < 0.65:
                    gestures.append("ASL Sign 'No' [Sign Language]")

        # 50. ASL Sign 'Please' [Sign Language]
        # Flat hand near chest
        if (t and i and m and r and p) or "Open Palm" in gestures:
            if pose_landmarks:
                # Left shoulder: 11, Right shoulder: 12
                # Chest center can be approximated by midpoint of shoulders
                mid_shoulder = {
                    'x': (pose_landmarks[11]['x'] + pose_landmarks[12]['x']) / 2,
                    'y': (pose_landmarks[11]['y'] + pose_landmarks[12]['y']) / 2,
                    'z': (pose_landmarks[11]['z'] + pose_landmarks[12]['z']) / 2
                }
                if dist3d(wrist, mid_shoulder) < 0.25:
                    gestures.append("ASL Sign 'Please' [Sign Language]")

        # 51. ASL Sign 'Sorry' [Sign Language]
        # Fist near chest
        if "Fist" in gestures:
            if pose_landmarks:
                mid_shoulder = {
                    'x': (pose_landmarks[11]['x'] + pose_landmarks[12]['x']) / 2,
                    'y': (pose_landmarks[11]['y'] + pose_landmarks[12]['y']) / 2,
                    'z': (pose_landmarks[11]['z'] + pose_landmarks[12]['z']) / 2
                }
                if dist3d(wrist, mid_shoulder) < 0.25:
                    gestures.append("ASL Sign 'Sorry' [Sign Language]")

        # 52. ASL Sign 'I Love You' [Sign Language]
        # Same as Spider-Man
        if t and i and p and not m and not r:
            gestures.append("ASL Sign 'I Love You' [Sign Language]")

        # 53. ASL Sign 'Father' [Sign Language]
        # Open hand, thumb touching forehead
        if (t and i and m and r and p) or "Open Palm" in gestures:
            if face_landmarks:
                forehead = face_landmarks[10]
                if dist3d(t_tip, forehead) < 0.18:
                    gestures.append("ASL Sign 'Father' [Sign Language]")
            elif pose_landmarks:
                head = pose_landmarks[0] # nose
                if dist3d(t_tip, head) < 0.22 and wrist['y'] < head['y']:
                    gestures.append("ASL Sign 'Father' [Sign Language]")

        # 54. ASL Sign 'Mother' [Sign Language]
        # Open hand, thumb touching chin
        if (t and i and m and r and p) or "Open Palm" in gestures:
            if face_landmarks:
                chin = face_landmarks[152]
                if dist3d(t_tip, chin) < 0.18:
                    gestures.append("ASL Sign 'Mother' [Sign Language]")
            elif pose_landmarks:
                mouth = pose_landmarks[0] # nose/mouth
                if dist3d(t_tip, mouth) < 0.25 and t_tip['y'] > mouth['y']:
                    gestures.append("ASL Sign 'Mother' [Sign Language]")

        # 55. ASL Sign 'Water' [Sign Language]
        # W-hand (index, middle, ring extended), index finger touching mouth/chin
        if i and m and r and not p:
            if face_landmarks:
                mouth = face_landmarks[13]
                if dist3d(i_tip, mouth) < 0.18 or dist3d(t_tip, mouth) < 0.18:
                    gestures.append("ASL Sign 'Water' [Sign Language]")
            elif pose_landmarks:
                mouth = pose_landmarks[0]
                if dist3d(i_tip, mouth) < 0.25:
                    gestures.append("ASL Sign 'Water' [Sign Language]")

        # 56. ASL Sign 'Eat' / 'Food' [Sign Language]
        # Closed flat O-hand near mouth
        if not i and not m and not r and not p:
            # semi closed O
            if dist3d(t_tip, i_tip) < 0.2 * hand_scale and dist3d(t_tip, m_tip) < 0.2 * hand_scale:
                if face_landmarks:
                    mouth = face_landmarks[13]
                    if dist3d(t_tip, mouth) < 0.18:
                        gestures.append("ASL Sign 'Eat' / 'Food' [Sign Language]")
                elif pose_landmarks:
                    mouth = pose_landmarks[0]
                    if dist3d(t_tip, mouth) < 0.25:
                        gestures.append("ASL Sign 'Eat' / 'Food' [Sign Language]")

        results[handedness] = list(set(gestures))

    # ----------------------------------------------------
    # Double Hand Gestures (4 Gestures)
    # ----------------------------------------------------
    if left_hand and right_hand:
        l_landmarks = left_hand['landmarks']
        r_landmarks = right_hand['landmarks']
        
        l_wrist = l_landmarks[0]
        r_wrist = r_landmarks[0]
        l_t_tip = l_landmarks[4]
        r_t_tip = r_landmarks[4]
        l_i_tip = l_landmarks[8]
        r_i_tip = r_landmarks[8]
        
        # 57. ASL Sign 'Help' [Sign Language]
        # Right hand is Thumbs Up, resting on left flat palm.
        # We check: left hand is Open Palm (or flat up), right hand is Thumbs Up, and their wrists/tips are very close.
        l_open = "Open Palm" in results['Left']
        r_thumbs_up = "Thumbs Up" in results['Right']
        if l_open and r_thumbs_up:
            if dist3d(r_wrist, l_wrist) < 0.25:
                results['DoubleHand'].append("ASL Sign 'Help' [Sign Language]")

        # 58. ASL Sign 'More' [Sign Language]
        # Both hands in "O-shapes" touching at fingertips
        l_o = "ASL Letter O [Sign Language]" in results['Left'] or "Fist" in results['Left']
        r_o = "ASL Letter O [Sign Language]" in results['Right'] or "Fist" in results['Right']
        if l_o and r_o:
            if dist3d(l_i_tip, r_i_tip) < 0.15:
                results['DoubleHand'].append("ASL Sign 'More' [Sign Language]")

        # 59. ASL Sign 'Stop' [Sign Language]
        # Right hand flat cuts down onto left hand flat palm
        l_open = "Open Palm" in results['Left']
        r_open = "Open Palm" in results['Right']
        if l_open and r_open:
            # hands close together
            if dist3d(l_wrist, r_wrist) < 0.2:
                results['DoubleHand'].append("ASL Sign 'Stop' [Sign Language]")

        # 60. ASL Sign 'Friend' [Sign Language]
        # Index fingers hook each other. Both index fingers extended, others curled, and index tips close.
        l_index = "Index Pointing Up" in results['Left'] or "Pistol" in results['Left']
        r_index = "Index Pointing Up" in results['Right'] or "Pistol" in results['Right']
        if l_index and r_index:
            if dist3d(l_i_tip, r_i_tip) < 0.12:
                results['DoubleHand'].append("ASL Sign 'Friend' [Sign Language]")

        # 61. Double Heart Hand
        # Both hands curved, left and right thumb tips close, index tips close, forming a heart.
        l_ratio = get_finger_states(l_landmarks)['index_ratio']
        r_ratio = get_finger_states(r_landmarks)['index_ratio']
        if 0.4 < l_ratio < 0.82 and 0.4 < r_ratio < 0.82:
            if dist3d(l_t_tip, r_t_tip) < 0.2 and dist3d(l_i_tip, r_i_tip) < 0.2:
                # index tips higher than thumb tips
                if l_i_tip['y'] < l_t_tip['y'] and r_i_tip['y'] < r_t_tip['y']:
                    results['DoubleHand'].append("Double Heart Hand")

    return results
