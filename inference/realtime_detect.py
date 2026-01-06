import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import joblib
import tensorflow as tf
import mediapipe as mp
from collections import deque

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

# Import utility functions
from utils.ear import eye_aspect_ratio
from utils.headpose import get_head_tilt

# Initialize MediaPipe Face Mesh
mp_face = mp.solutions.face_mesh
fm = mp_face.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Load models
print("[INFO] Loading models...")
ml_model = joblib.load("models/ml_model.pkl")
lstm_model = tf.keras.models.load_model("models/lstm_model.h5")
print("[INFO] Models loaded successfully!")

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Initialize buffer and counters
buffer = deque(maxlen=30)  # More efficient than list
closed_frames = 0
frame_count = 0
alarm_playing = False

# Eye landmark indices (MediaPipe)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

print("\n[INFO] Real-time drowsiness detection started!")
print("[INFO] Press 'Q' or 'ESC' to exit.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("[WARNING] Failed to read frame")
        break
    
    frame_count += 1
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = fm.process(rgb)
    
    status = "Alert"
    color = (0, 255, 0)  # Green for alert
    ear_value = 0.0
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape
        
        # Extract eye landmarks
        left_eye_pts = np.array([
            [landmarks.landmark[i].x * w, landmarks.landmark[i].y * h]
            for i in LEFT_EYE
        ])
        right_eye_pts = np.array([
            [landmarks.landmark[i].x * w, landmarks.landmark[i].y * h]
            for i in RIGHT_EYE
        ])
        
        # Calculate EAR
        left_ear = eye_aspect_ratio(left_eye_pts)
        right_ear = eye_aspect_ratio(right_eye_pts)
        ear_value = (left_ear + right_ear) / 2.0
        
        # Get head tilt
        tilt = get_head_tilt(landmarks, w, h)
        
        # Add to buffer
        buffer.append(ear_value)
        
        # Make predictions when buffer is full
        if len(buffer) == 30:
            # ML Model prediction (using mean and std of buffer)
            ml_features = np.array([[np.mean(buffer), np.std(buffer)]])
            ml_pred = ml_model.predict(ml_features)[0]
            
            # LSTM Model prediction
            lstm_seq = np.array(buffer).reshape(1, 30, 1).astype(np.float32)
            lstm_prob = lstm_model.predict(lstm_seq, verbose=0)[0][0]
            lstm_pred = int(lstm_prob > 0.5)
            
            # Combine predictions (either model detects drowsiness)
            if ml_pred == 1 or lstm_pred == 1:
                status = "DROWSY"
                color = (0, 0, 255)  # Red for drowsy
                closed_frames += 1
            else:
                status = "Alert"
                color = (0, 255, 0)  # Green for alert
                closed_frames = 0
            
            # Trigger alarm if drowsy for extended period
            if closed_frames > 15 and not alarm_playing:
                alarm_playing = True
                try:
                    # Try to play alarm sound
                    import winsound
                    winsound.Beep(2500, 500)
                    # If alarm.wav exists, play it
                    if Path("assets/alarm.wav").exists():
                        winsound.PlaySound("assets/alarm.wav", winsound.SND_ASYNC)
                except:
                    print("[WARNING] Could not play alarm sound")
            elif closed_frames <= 5:
                alarm_playing = False
    
    # Draw overlay
    overlay = frame.copy()
    
    # Status box
    cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Status text
    cv2.putText(frame, f"Status: {status}", (20, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"EAR: {ear_value:.3f}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Frames: {closed_frames}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Warning if drowsy
    if status == "DROWSY":
        cv2.putText(frame, "!!! WAKE UP !!!", (frame.shape[1]//2 - 150, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    
    # Display FPS
    if frame_count > 30:
        cv2.putText(frame, "Press Q to exit", (frame.shape[1] - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Show frame
    cv2.imshow("BlinkWatch - Drowsiness Detection", frame)
    
    # Check for exit
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q') or key == ord('Q'):  # ESC or Q
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
fm.close()
print("\n[INFO] Webcam closed. System exited safely.")