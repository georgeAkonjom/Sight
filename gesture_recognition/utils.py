import math

def dist3d(p1, p2):
    """Calculate the 3D Euclidean distance between two points (dicts with 'x', 'y', 'z')."""
    return math.sqrt((p1['x'] - p2['x'])**2 + 
                     (p1['y'] - p2['y'])**2 + 
                     (p1['z'] - p2['z'])**2)

def dist2d(p1, p2):
    """Calculate the 2D Euclidean distance in the x-y plane."""
    return math.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)

def get_angle(p1, p2, p3):
    """
    Calculate the angle (in degrees) at p2 formed by vectors p2->p1 and p2->p3.
    p1, p2, p3 are dicts with 'x', 'y', 'z'.
    """
    v1 = {
        'x': p1['x'] - p2['x'],
        'y': p1['y'] - p2['y'],
        'z': p1['z'] - p2['z']
    }
    v2 = {
        'x': p3['x'] - p2['x'],
        'y': p3['y'] - p2['y'],
        'z': p3['z'] - p2['z']
    }
    
    dot = v1['x']*v2['x'] + v1['y']*v2['y'] + v1['z']*v2['z']
    len1 = math.sqrt(v1['x']**2 + v1['y']**2 + v1['z']**2)
    len2 = math.sqrt(v2['x']**2 + v2['y']**2 + v2['z']**2)
    
    if len1 * len2 == 0:
        return 0.0
    
    cos_angle = dot / (len1 * len2)
    # Clamp for floating-point inaccuracies
    cos_angle = max(-1.0, min(1.0, cos_angle))
    
    return math.degrees(math.acos(cos_angle))

def get_finger_states(landmarks):
    """
    Determine whether each of the 5 fingers is extended.
    landmarks: list of 21 landmark dicts.
    Returns: dict with keys: 'thumb', 'index', 'middle', 'ring', 'pinky' -> bool
    and curl ratios.
    """
    # Hand scale: wrist (0) to middle finger MCP (9)
    hand_scale = dist3d(landmarks[0], landmarks[9])
    if hand_scale == 0:
        hand_scale = 0.001

    # Finger Tip to MCP distance compared to fully open segments
    # Index: 5 (MCP), 6 (PIP), 7 (DIP), 8 (Tip)
    index_seg_sum = dist3d(landmarks[6], landmarks[5]) + dist3d(landmarks[7], landmarks[6]) + dist3d(landmarks[8], landmarks[7])
    index_ratio = dist3d(landmarks[8], landmarks[5]) / max(0.001, index_seg_sum)
    index_extended = index_ratio > 0.82

    # Middle: 9 (MCP), 10 (PIP), 11 (DIP), 12 (Tip)
    middle_seg_sum = dist3d(landmarks[10], landmarks[9]) + dist3d(landmarks[11], landmarks[10]) + dist3d(landmarks[12], landmarks[11])
    middle_ratio = dist3d(landmarks[12], landmarks[9]) / max(0.001, middle_seg_sum)
    middle_extended = middle_ratio > 0.82

    # Ring: 13 (MCP), 14 (PIP), 15 (DIP), 16 (Tip)
    ring_seg_sum = dist3d(landmarks[14], landmarks[13]) + dist3d(landmarks[15], landmarks[14]) + dist3d(landmarks[16], landmarks[15])
    ring_ratio = dist3d(landmarks[16], landmarks[13]) / max(0.001, ring_seg_sum)
    ring_extended = ring_ratio > 0.82

    # Pinky: 17 (MCP), 18 (PIP), 19 (DIP), 20 (Tip)
    pinky_seg_sum = dist3d(landmarks[18], landmarks[17]) + dist3d(landmarks[19], landmarks[18]) + dist3d(landmarks[20], landmarks[19])
    pinky_ratio = dist3d(landmarks[20], landmarks[17]) / max(0.001, pinky_seg_sum)
    pinky_extended = pinky_ratio > 0.82

    # Thumb: 0 (Wrist), 1 (CMC), 2 (MCP), 3 (IP), 4 (Tip)
    # Thumb moves differently. We check its angle and distance from index MCP (5) and wrist (0).
    # If the thumb tip (4) is close to the index MCP (5) or the palm, it is folded.
    # If it is spread away, it's extended.
    thumb_ratio = dist3d(landmarks[4], landmarks[2]) / max(0.001, dist3d(landmarks[3], landmarks[2]) + dist3d(landmarks[4], landmarks[3]))
    # Additionally, check distance from thumb tip (4) to middle MCP (9) and index MCP (5)
    thumb_to_index_mcp = dist3d(landmarks[4], landmarks[5]) / hand_scale
    # If it's too close to the side of the hand, it is folded.
    # In ASL letters (like A, S, T, M, N), the thumb is tucked in. In open hand, it's out.
    thumb_extended = thumb_ratio > 0.85 and thumb_to_index_mcp > 0.45

    return {
        'thumb': thumb_extended,
        'index': index_extended,
        'middle': middle_extended,
        'ring': ring_extended,
        'pinky': pinky_extended,
        'thumb_ratio': thumb_ratio,
        'index_ratio': index_ratio,
        'middle_ratio': middle_ratio,
        'ring_ratio': ring_ratio,
        'pinky_ratio': pinky_ratio,
        'thumb_to_index_mcp': thumb_to_index_mcp,
        'hand_scale': hand_scale
    }
