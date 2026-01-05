# import cv2
# import numpy as np

# def get_head_tilt(face_landmarks, w, h):
#     nose = np.array([face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h])
#     chin = np.array([face_landmarks.landmark[152].x * w, face_landmarks.landmark[152].y * h])
#     diff = chin - nose
#     return np.degrees(np.arctan2(diff[1], diff[0]))


import numpy as np

def get_head_tilt(landmarks, img_w, img_h):
    """
    Calculate head tilt angle from facial landmarks
    
    Args:
        landmarks: MediaPipe face landmarks
        img_w: image width
        img_h: image height
        
    Returns:
        float: head tilt angle in degrees
    """
    # Get nose tip and chin landmarks
    nose_tip = landmarks.landmark[1]
    chin = landmarks.landmark[152]
    
    # Convert to pixel coordinates
    nose_y = nose_tip.y * img_h
    chin_y = chin.y * img_h
    
    # Calculate vertical distance (simple tilt approximation)
    vertical_dist = abs(chin_y - nose_y)
    
    # Normalize to angle (simplified)
    tilt_angle = vertical_dist / img_h * 90
    
    return tilt_angle