import cv2
import mediapipe as mp
from ultralytics import YOLO

# === Load YOLOv8 model ===
yolo_model = YOLO("runs/detect/train5/weights/best.pt")  # ganti dengan path model kamu

# === Init MediaPipe Hands ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# === Open Webcam ===
cap = cv2.VideoCapture(0)

def normalize_box(box, frame_w, frame_h):
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_w, x2)
    y2 = min(frame_h, y2)
    return x1, y1, x2, y2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_h, frame_w, _ = frame.shape

    # === YOLO Inference ===
    results = yolo_model(frame_rgb)[0]

    hand_detected = False

    # === Loop setiap hasil YOLO ===
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if cls == 0 and conf > 0.5:  # Kelas tangan + confidence > 0.5
            hand_detected = True
            x1, y1, x2, y2 = normalize_box(box.xyxy[0], frame_w, frame_h)

            # Crop area tangan dan resize untuk MediaPipe
            hand_roi = frame[y1:y2, x1:x2]
            if hand_roi.size == 0:
                continue
            hand_roi_resized = cv2.resize(hand_roi, (224, 224))
            hand_rgb = cv2.cvtColor(hand_roi_resized, cv2.COLOR_BGR2RGB)
            mp_results = hands.process(hand_rgb)

            if mp_results.multi_hand_landmarks:
                for hand_landmarks in mp_results.multi_hand_landmarks:
                    # Gambar landmark di ROI yang diresize
                    mp_drawing.draw_landmarks(hand_roi_resized, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Tampilkan hasil ke frame utama
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Hand ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Hand ROI", hand_roi_resized)

    # === Fallback ke MediaPipe langsung jika tidak terdeteksi oleh YOLO ===
    if not hand_detected:
        mp_results = hands.process(frame_rgb)
        if mp_results.multi_hand_landmarks:
            for hand_landmarks in mp_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, "Fallback: MediaPipe Full Frame", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # === Tampilkan frame utama ===
    cv2.imshow("YOLO + MediaPipe Hands", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC untuk keluar
        break

hands.close()
cap.release()
cv2.destroyAllWindows()
