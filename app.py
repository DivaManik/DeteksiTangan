import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

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
    "10": "SEPULUH"
}

# Cek apakah tangan dibalik
def is_hand_flipped(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    thumb_mcp = hand_landmarks.landmark[2]
    return abs(thumb_tip.z - thumb_mcp.z) > 0.1 and thumb_tip.z > thumb_mcp.z

# Dapatkan status jari terbuka/tutup
def get_finger_states(hand_landmarks):
    finger_states = []

    # Thumb horizontal
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    wrist = hand_landmarks.landmark[0]
    is_thumb_open = thumb_tip.x < thumb_ip.x if thumb_tip.y < wrist.y else thumb_tip.x > thumb_ip.x
    finger_states.append(is_thumb_open)

    # Fingers vertical
    tips = [8, 12, 16, 20]
    pip_joints = [6, 10, 14, 18]
    for tip, pip in zip(tips, pip_joints):
        tip_y = hand_landmarks.landmark[tip].y
        pip_y = hand_landmarks.landmark[pip].y
        is_open = tip_y < pip_y
        finger_states.append(is_open)

    return finger_states  # thumb, index, middle, ring, pinky

# Hitung jarak antara dua landmark
def calculate_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# Deteksi gesture berdasarkan jari dan posisi
def detect_gesture(hand_landmarks, finger_states):
    thumb, index, middle, ring, pinky = finger_states

    # Gesture 9: thumb & index berdekatan, middle, ring, pinky terbuka
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

# Deteksi gesture silang (angka 10)
def detect_cross_gesture(results):
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        l_wrist = results.multi_hand_landmarks[0].landmark[0]
        r_wrist = results.multi_hand_landmarks[1].landmark[0]
        dx = abs(l_wrist.x - r_wrist.x)
        dy = abs(l_wrist.y - r_wrist.y)
        if dx < 0.1 and dy < 0.1:
            return "10"
    return None

# Main loop
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        detected_gesture = None
        flip_warning = False

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if is_hand_flipped(hand_landmarks):
                    flip_warning = True
                    break

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                finger_states = get_finger_states(hand_landmarks)
                gesture = detect_gesture(hand_landmarks, finger_states)
                if gesture:
                    detected_gesture = gesture

        if not detected_gesture and not flip_warning:
            detected_gesture = detect_cross_gesture(results)

        if flip_warning:
            cv2.putText(frame, "Tidak bisa membalikkan tangan!", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        elif detected_gesture:
            text = f"{detected_gesture} - {gesture_sibi[detected_gesture]}"
            cv2.putText(frame, text, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("Gesture SIBI", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
