# import numpy as np

# def eye_aspect_ratio(eye):
#     A = np.linalg.norm(eye[1] - eye[5])
#     B = np.linalg.norm(eye[2] - eye[4])
#     C = np.linalg.norm(eye[0] - eye[3])
#     return (A + B) / (2.0 * C)



import numpy as np

def eye_aspect_ratio(eye_points):
    """
    Calculate Eye Aspect Ratio (EAR)
    
    Args:
        eye_points: numpy array of 6 (x,y) coordinates
        
    Returns:
        float: EAR value
    """
    # Compute vertical distances
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    
    # Compute horizontal distance
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    
    # Calculate EAR
    ear = (A + B) / (2.0 * C)
    
    return ear