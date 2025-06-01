import cv2
import mediapipe as mp
import math
from ultralytics import YOLO

# Load YOLO model
yolo_model = YOLO("runs/detect/train5/weights/best.pt")  # Ganti path jika perlu

# Inisialisasi MediaPipe
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

# Mapping gesture ke teks
gesture_sibi = {
    "1": "SATU",
    "2": "DUA",
    "3": "TIGA",
    "4": "EMPAT",
    "5": "LIMA",
    "6": "ENAM",
    "7": "TUJUH",
    "8": "DELAPAN",
    "9": "SEMBILAN",
    "10": "SEPULUH",
    "HALLO": "HALLO"
}

def calculate_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def get_finger_states(hand_landmarks, handedness):
    finger_states = []
    if handedness == "Right":
        thumb_is_open = hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x
    else:
        thumb_is_open = hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x
    finger_states.append(thumb_is_open)

    tips = [8, 12, 16, 20]
    pip_joints = [6, 10, 14, 18]
    for tip, pip in zip(tips, pip_joints):
        is_open = hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y
        finger_states.append(is_open)

    return finger_states

def detect_gesture(finger_states, hand_landmarks):
    thumb, index, middle, ring, pinky = finger_states
    dist = calculate_distance(hand_landmarks.landmark[4], hand_landmarks.landmark[8])
    if dist < 0.05 and middle and ring and pinky:
        return "9"
    if index and not thumb and not middle and not ring and not pinky:
        return "1"
    elif index and middle and not thumb and not ring and not pinky:
        return "2"
    elif thumb and index and middle and not ring and not pinky:
        return "3"
    elif index and middle and ring and pinky and not thumb:
        return "4"
    elif thumb and index and middle and ring and pinky:
        return "5"
    elif index and middle and ring and not thumb and not pinky:
        return "6"
    elif index and middle and pinky and not thumb and not ring:
        return "7"
    elif index and ring and pinky and not thumb and not middle:
        return "8"
    else:
        return None

def detect_cross_gesture(results):
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        left_wrist = results.multi_hand_landmarks[0].landmark[0]
        right_wrist = results.multi_hand_landmarks[1].landmark[0]
        dx = abs(left_wrist.x - right_wrist.x)
        dy = abs(left_wrist.y - right_wrist.y)
        if dx < 0.1 and dy < 0.1:
            return "10"
    return None

def detect_ten_gesture_by_thumb(finger_states):
    thumb, index, middle, ring, pinky = finger_states
    if thumb and not index and not middle and not ring and not pinky:
        return "10"
    return None

def detect_hallo_with_face(hand_landmarks, face_landmarks):
    tips = [4, 8, 12, 16, 20]
    tip_points = [hand_landmarks.landmark[i] for i in tips]
    pip_joints = [3, 6, 10, 14, 18]

    is_open = [hand_landmarks.landmark[t].y < hand_landmarks.landmark[p].y for t, p in zip(tips, pip_joints)]
    if not all(is_open):
        return False

    for i in range(len(tip_points) - 1):
        if calculate_distance(tip_points[i], tip_points[i + 1]) > 0.1:
            return False

    avg_x = sum(p.x for p in tip_points) / len(tip_points)
    avg_y = sum(p.y for p in tip_points) / len(tip_points)

    left_temple = face_landmarks.landmark[127]
    right_temple = face_landmarks.landmark[356]

    d_left = math.sqrt((avg_x - left_temple.x)**2 + (avg_y - left_temple.y)**2)
    d_right = math.sqrt((avg_x - right_temple.x)**2 + (avg_y - right_temple.y)**2)

    return d_left < 0.07 or d_right < 0.07

# Mulai kamera
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
) as hands, mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5
) as face_mesh:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)
        face_results = face_mesh.process(rgb_frame)

        detected_gesture = None

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                handedness = handedness_info.classification[0].label
                finger_states = get_finger_states(hand_landmarks, handedness)
                gesture = detect_gesture(finger_states, hand_landmarks)

                if gesture:
                    detected_gesture = gesture
                else:
                    gesture = detect_ten_gesture_by_thumb(finger_states)
                    if gesture:
                        detected_gesture = gesture

                if face_results.multi_face_landmarks:
                    face_landmarks = face_results.multi_face_landmarks[0]
                    if detect_hallo_with_face(hand_landmarks, face_landmarks):
                        detected_gesture = "HALLO"

        if not detected_gesture:
            gesture_10 = detect_cross_gesture(results)
            if gesture_10:
                detected_gesture = gesture_10

        # Jika gesture tetap tidak terdeteksi → jalankan YOLO
        if not detected_gesture:
            yolo_results = yolo_model.predict(frame, conf=0.5, verbose=False)
            for result in yolo_results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    label = yolo_model.names[cls]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 100, 0), 2)

        # Tampilkan gesture jika ada
        if detected_gesture:
            text = f"{detected_gesture} - {gesture_sibi[detected_gesture]}"
            cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # Jangan tampilkan landmark (MediaPipe hanya proses internal)
        cv2.imshow('Gesture SIBI + YOLO', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC untuk keluar
            break

cap.release()
cv2.destroyAllWindows()
