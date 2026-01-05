from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolov8n.pt")

video_path = "sample_video/farm_cctv.mp4"
cap = cv2.VideoCapture(video_path)

prev_centers = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, None, fx=1.5, fy=1.5)

    results = model(frame, imgsz=1280, conf=0.15)

    cow_centers = []
    cow_count = 0
    motion_score = 0.0

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            if label == "cow":
                cow_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                cow_centers.append((cx, cy))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    # Hareket hesabı
    if prev_centers:
        for c, p in zip(cow_centers, prev_centers):
            motion_score += np.linalg.norm(np.array(c) - np.array(p))

    prev_centers = cow_centers.copy()

    # Bilgi yazdır
    cv2.putText(frame, f"Cow Count: {cow_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(frame, f"Motion Score: {motion_score:.2f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Farm Monitoring Demo", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
