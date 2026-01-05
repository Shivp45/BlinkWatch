import sys
import os
sys.path.append(os.getcwd())  # Ensure project root is in path

import cv2
import pandas as pd
import mediapipe as mp
import numpy as np

from utils.ear import eye_aspect_ratio
from utils.headpose import get_head_tilt  # Function name now matches util file

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True, max_num_faces=1)
cap = cv2.VideoCapture(0)

frame_count = 0
records = []

print("\n[INFO] Webcam opened for recording. Press ESC when done.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Webcam frame not received.")
        break

    frame_count += 1
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        fl = results.multi_face_landmarks[0]
        h, w, _ = frame.shape

        left_idx  = [33, 160, 158, 133, 153, 144]
        right_idx = [362, 385, 387, 263, 373, 380]

        left_eye  = np.array([[fl.landmark[i].x * w, fl.landmark[i].y * h] for i in left_idx])
        right_eye = np.array([[fl.landmark[i].x * w, fl.landmark[i].y * h] for i in right_idx])

        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        tilt = get_head_tilt(fl, w, h)

        records.append([ear, tilt, frame_count])

    cv2.putText(frame, f"Frames Captured: {frame_count}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.imshow("Recording Blink Data", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# --- Append CSV instead of overwrite ---
file = os.path.join("dataset", "features.csv")
df = pd.DataFrame(records, columns=["EAR", "Head_Tilt", "Frame"])

if os.path.exists(file):
    df.to_csv(file, mode='a', header=False, index=False)
else:
    df.to_csv(file, index=False)

print("\n[INFO] Recording stopped.")
print("[INFO] Total Frames Captured:", frame_count)
print("[INFO] Data saved to dataset/features.csv\n")
