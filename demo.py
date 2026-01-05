from ultralytics import YOLO
import cv2
import numpy as np

# -----------------------------
# Model ve Video Ayarları
# -----------------------------
MODEL_PATH = "yolov8n.pt"
VIDEO_PATH = "test_video/frame.mp4"

IMG_SIZE = 1280
CONF_TH = 0.10
IOU_TH = 0.4
COW_CLASS_ID = 19  # COCO cow

# -----------------------------
# Model Yükle
# -----------------------------
model = YOLO(MODEL_PATH)

# -----------------------------
# Video Aç
# -----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Video açılamadı")

# Önceki frame merkezleri
prev_centers = []

# -----------------------------
# Ana Döngü
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Uzaktan çekimler için upscale
    frame = cv2.resize(frame, None, fx=1.5, fy=1.5)

    # YOLO inference
    results = model(
        frame,
        imgsz=IMG_SIZE,
        conf=CONF_TH,
        iou=IOU_TH,
        classes=[COW_CLASS_ID]
    )

    cow_centers = []
    cow_count = 0
    motion_score = 0.0

    # -----------------------------
    # Tespitler
    # -----------------------------
    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf_score = float(box.conf[0])

            # Merkez noktası
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            cow_centers.append((cx, cy))
            cow_count += 1

            # Çizimler
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            cv2.putText(
                frame,
                f"cow {conf_score:.2f}",
                (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

    # -----------------------------
    # Hareket (Aktivite) Skoru
    # -----------------------------
    if prev_centers and cow_centers:
        min_len = min(len(prev_centers), len(cow_centers))
        for i in range(min_len):
            motion_score += np.linalg.norm(
                np.array(cow_centers[i]) - np.array(prev_centers[i])
            )

    prev_centers = cow_centers.copy()

    # -----------------------------
    # Bilgi Overlay
    # -----------------------------
    cv2.rectangle(frame, (10, 10), (380, 120), (0, 0, 0), -1)

    cv2.putText(frame, f"Cow Count: {cow_count}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.putText(frame, f"Motion Score: {motion_score:.2f}", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.imshow("Farm Animal Monitoring Demo", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

# -----------------------------
# Temizlik
# -----------------------------
cap.release()
cv2.destroyAllWindows()
