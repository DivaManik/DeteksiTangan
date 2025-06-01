import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

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
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    dist = calculate_distance(thumb_tip, index_tip)
    if dist < 0.05 and middle and ring and pinky:
        return "9"

    if not thumb and index and not middle and not ring and not pinky:
        return "1"
    elif not thumb and index and middle and not ring and not pinky:
        return "2"
    elif thumb and index and middle and not ring and not pinky:
        return "3"
    elif not thumb and index and middle and ring and pinky:
        return "4"
    elif thumb and index and middle and ring and pinky:
        return "5"
    elif not thumb and index and middle and ring and not pinky:
        return "6"
    elif not thumb and index and middle and not ring and pinky:
        return "7"
    elif not thumb and index and not middle and ring and pinky:
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

    is_open = []
    pip_joints = [3, 6, 10, 14, 18]
    for tip, pip in zip(tips, pip_joints):
        is_open.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y)
    if not all(is_open):
        return False

    for i in range(len(tip_points) - 1):
        if calculate_distance(tip_points[i], tip_points[i + 1]) > 0.1:
            return False

    # Hitung rata-rata ujung jari
    avg_x = sum(p.x for p in tip_points) / len(tip_points)
    avg_y = sum(p.y for p in tip_points) / len(tip_points)

    # Landmark pelipis: 127 (kiri), 356 (kanan)
    left_temple = face_landmarks.landmark[127]
    right_temple = face_landmarks.landmark[356]

    d_left = math.sqrt((avg_x - left_temple.x)**2 + (avg_y - left_temple.y)**2)
    d_right = math.sqrt((avg_x - right_temple.x)**2 + (avg_y - right_temple.y)**2)

    if d_left < 0.07 or d_right < 0.07:
        return True

    return False

# Main loop
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
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                handedness = handedness_info.classification[0].label

                finger_states = get_finger_states(hand_landmarks, handedness)
                gesture = detect_gesture(finger_states, hand_landmarks)
                if gesture:
                    detected_gesture = gesture
                else:
                    gesture = detect_ten_gesture_by_thumb(finger_states)
                    if gesture:
                        detected_gesture = gesture

                # Cek gesture "HALLO"
                if face_results.multi_face_landmarks:
                    face_landmarks = face_results.multi_face_landmarks[0]
                    if detect_hallo_with_face(hand_landmarks, face_landmarks):
                        detected_gesture = "HALLO"

        if not detected_gesture:
            gesture_10 = detect_cross_gesture(results)
            if gesture_10:
                detected_gesture = gesture_10

        if detected_gesture:
            text = f"{detected_gesture} - {gesture_sibi[detected_gesture]}"
            cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # Tambahkan visualisasi face landmarks (opsional)
        if face_results.multi_face_landmarks:
            for face_landmark in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(frame, face_landmark, mp_face_mesh.FACEMESH_TESSELATION)

        cv2.imshow('Gesture SIBI', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
