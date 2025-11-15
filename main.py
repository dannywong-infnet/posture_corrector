import cv2
import mediapipe as mp
import numpy as np
from playsound3 import playsound
import time
import os

# --- HELPER FUNCTIONS ---
def calculate_angle(a, b, c):
    """Calculates the angle (in degrees) between three points a, b, and c.
        Point 'b' is the vertex of the angle."""
    
    if not all(isinstance(p, tuple) and len(p) == 2 for p in [a, b, c]):
        return 0
    
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    
    if norm_ba == 0 or norm_bc == 0:
        return 0

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    angle_deg = np.degrees(angle_rad)
    return angle_deg

def draw_angle(image, a, b, c, angle_value, color):
    """Draws lines and text representing the angle on the frame."""
    cv2.line(image, a, b, color, 3)
    cv2.line(image, b, c, color, 3)
    cv2.circle(image, a, 5, color, cv2.FILLED)
    cv2.circle(image, b, 5, color, cv2.FILLED)
    cv2.circle(image, c, 5, color, cv2.FILLED)
    
    text_pos = (b[0] + 15, b[1] - 15)
    cv2.putText(image, f"{angle_value:.1f}", text_pos, 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

# --- INITIALIZATION ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
# Increased confidence for tracking stability
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)
cap = cv2.VideoCapture(1)  

if not cap.isOpened():
    print("FATAL ERROR: Could not open video stream.")
    exit()

# Variables for calibration
is_calibrated = False
calibration_frames = 0
calibration_shoulder_angles = []
calibration_y_distances = [] 
calibration_x_distances = [] 

# Thresholds (calibrated in the loop)
shoulder_threshold_max = 0 
distance_y_threshold_min = 0 # If Y distance drops below this, it's a slouch
distance_x_threshold_max = 0 # If X distance exceeds this, it's a lateral lean

# Variables for alerts
last_alert_time = time.time()
alert_cooldown = 15  # seconds between alerts
sound_file = 'alert.mp3'  
if not os.path.exists(sound_file):
    print(f"WARNING: Sound file '{sound_file}' not found. No sound alert will play.")


# --- MAIN LOOP ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        continue

    # Flip the frame for a 'mirror' view
    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape

    # Convert the BGR image to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame
    results = pose.process(rgb_frame)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # STEP 2: Pose Detection (Coordinate Extraction)
        nose = (int(landmarks[mp_pose.PoseLandmark.NOSE.value].x * W),
                int(landmarks[mp_pose.PoseLandmark.NOSE.value].y * H))
        left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * W),
                         int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * H))
        right_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * W),
                          int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * H))
        
        # Calculate the midpoint between the shoulders to represent the body's vertical axis
        shoulder_midpoint_x = int((left_shoulder[0] + right_shoulder[0]) / 2)
        shoulder_midpoint = (shoulder_midpoint_x, left_shoulder[1])


        # STEP 3: Angle and Distance Calculation
        
        # 1. Shoulder Angle (Tilt)
        shoulder_angle = calculate_angle(left_shoulder, right_shoulder, (right_shoulder[0] + 50, right_shoulder[1]))
        
        # 2. Head-Shoulder Y Distance (Vertical Slouching)
        # Measures the vertical space between the head (Nose) and the body (Shoulder).
        head_shoulder_y_distance = abs(left_shoulder[1] - nose[1])

        # 3. Head-Shoulder X Distance (Lateral Slouching)
        # Measures the horizontal distance between the head (Nose) and the body's vertical center (Shoulder Midpoint X).
        head_shoulder_x_distance = abs(nose[0] - shoulder_midpoint_x)


        # STEP 1: Calibration
        if not is_calibrated and calibration_frames < 60: # Use 60 frames for stability
            if all(p is not None for p in [left_shoulder, right_shoulder, nose]):
                calibration_shoulder_angles.append(shoulder_angle)
                calibration_y_distances.append(head_shoulder_y_distance)
                calibration_x_distances.append(head_shoulder_x_distance)
                calibration_frames += 1
            cv2.putText(frame, f"Calibrating... {calibration_frames}/60", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            
        elif not is_calibrated and calibration_frames >= 60:
            # Threshold 1: Shoulder Tilt (Max acceptable angle)
            shoulder_threshold_max = np.mean(calibration_shoulder_angles) + 2.0 
            
            # Threshold 2: Y Distance (Min acceptable distance, margin of 25 pixels)
            distance_y_threshold_min = np.mean(calibration_y_distances) - 20
            
            # Threshold 3: X Distance (Max acceptable distance, margin of 15 pixels)
            distance_x_threshold_max = np.mean(calibration_x_distances) + 15 
            
            is_calibrated = True
            print(f"Calibration complete.")
            print(f"   Max Shoulder Tilt: {shoulder_threshold_max:.1f}°")
            print(f"   Min Y Distance (Slouch): {distance_y_threshold_min:.1f} pixels")
            print(f"   Max X Distance (Lateral): {distance_x_threshold_max:.1f} pixels")

        # Draw skeleton and landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Draw visualizations
        # 1. Shoulder Tilt Angle
        draw_angle(frame, left_shoulder, right_shoulder, (right_shoulder[0] + 50, right_shoulder[1]), 
                   shoulder_angle, (255, 0, 0)) # Blue
        
        # 2. Y Distance (Vertical line for slouch check)
        cv2.line(frame, nose, (nose[0], left_shoulder[1]), (0, 255, 255), 2)
        
        # 3. X Distance (Horizontal line for lateral check)
        cv2.line(frame, nose, (shoulder_midpoint[0], nose[1]), (255, 255, 0), 2)


        # STEP 4: Feedback
        if is_calibrated:
            current_time = time.time()
            
            # Posture check conditions:
            poor_shoulder_tilt = (shoulder_angle > shoulder_threshold_max)
            poor_y_distance = (head_shoulder_y_distance < distance_y_threshold_min)
            poor_x_distance = (head_shoulder_x_distance > distance_x_threshold_max)

            is_poor_posture = poor_shoulder_tilt or poor_y_distance or poor_x_distance

            if is_poor_posture:
                status = "Poor Posture"
                color = (0, 0, 255)  # Red
                if current_time - last_alert_time > alert_cooldown:
                    print("Poor posture detected! Please sit up straight.")
                    if os.path.exists(sound_file):
                        playsound(sound_file) 
                    last_alert_time = current_time
            else:
                status = "Good Posture"
                color = (0, 255, 0)  # Green

            # Display status and angle/distance values
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"Shoulder Tilt: {shoulder_angle:.1f} / <{shoulder_threshold_max:.1f}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Y Dist (Slouch): {head_shoulder_y_distance:.1f} / >{distance_y_threshold_min:.1f}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"X Dist (Lateral): {head_shoulder_x_distance:.1f} / <{distance_x_threshold_max:.1f}", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Posture Corrector', frame)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- CLEANUP ---
cap.release()
cv2.destroyAllWindows()
pose.close()