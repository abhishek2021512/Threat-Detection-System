import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import face_recognition
import pygame
import time
import os

# Initialize YOLOv8 model for object detection
model = YOLO("yolov8s.pt")
threat_objects = ["knife", "smoke", "needle", "scissors", "chainsaw", "gun"]

# Set up MediaPipe Hands for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Load known face encodings and names (example data)
known_face_encodings = [
    face_recognition.face_encodings(face_recognition.load_image_file("img.jpg"))[0]
]
known_face_names = ["Abhishek Dhawan"]

# Initialize webcam
cap = cv2.VideoCapture(0)

# Variables for face detection smoothing
smooth_factor = 10
x_history = []
y_history = []

# Variables for hand movement detection
previous_hand_positions = []
movement_threshold = 50

# Siren sound configuration
siren_sound = "siren.mp3"  # Ensure this is the correct path to your 20-second siren audio file
if not os.path.exists(siren_sound):
    print(f"Error: Audio file '{siren_sound}' not found. Please check the path.")
    exit()

# Initialize pygame mixer for audio playback
pygame.mixer.init()
siren = pygame.mixer.Sound(siren_sound)

# Variable to track the last time the siren was played
last_siren_start_time = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Reset threat detection flag for each frame
    threat_detected = False

    # Get frame dimensions for centering
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2

    # Face detection with smoothing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        x_history.append(face_center_x)
        y_history.append(face_center_y)
        if len(x_history) > smooth_factor:
            x_history.pop(0)
            y_history.pop(0)
        smooth_x = int(sum(x_history) / len(x_history))
        smooth_y = int(sum(y_history) / len(y_history))
        offset_x = smooth_x - center_x
        offset_y = smooth_y - center_y
        cv2.putText(frame, f"Offset: ({offset_x}, {offset_y})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (smooth_x - 5, smooth_y - 5), (smooth_x + 5, smooth_y + 5), (255, 0, 0), -1)

    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

    # Hand detection and movement tracking
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_result = hands.process(frame_rgb)

    current_hand_positions = []
    if hand_result.multi_hand_landmarks:
        for hand_landmarks in hand_result.multi_hand_landmarks:
            wrist = hand_landmarks.landmark[0]
            h, w, _ = frame.shape
            wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
            current_hand_positions.append((wrist_x, wrist_y))

    if previous_hand_positions:
        for i, current_pos in enumerate(current_hand_positions):
            if i < len(previous_hand_positions):
                prev_pos = previous_hand_positions[i]
                movement = np.linalg.norm(np.array(prev_pos) - np.array(current_pos))
                if movement > movement_threshold:
                    threat_detected = True
                    cv2.putText(frame, "Suspicious Movement!", (current_pos[0], current_pos[1] - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    previous_hand_positions = current_hand_positions

    # Object detection with YOLOv8
    results = model(frame, verbose=False)
    for result in results:
        for det in result.boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            conf = float(det.conf)
            class_id = int(det.cls[0])
            label = model.names[class_id]
            if conf > 0.3:
                if label in threat_objects:
                    threat_detected = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"THREAT: {label}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                elif label != "person":
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Facial recognition (not used for threat detection here)
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Siren playback logic using pygame
    if threat_detected:
        current_time = time.time()
        if last_siren_start_time is None or current_time - last_siren_start_time >= 120:  # 2 minutes = 120 seconds
            siren.play()  # Play the siren in the background (20 seconds)
            last_siren_start_time = current_time
    else:
        last_siren_start_time = None

    # Display the processed frame
    cv2.imshow("Threat Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()