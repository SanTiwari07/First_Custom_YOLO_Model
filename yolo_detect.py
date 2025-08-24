import cv2
from ultralytics import YOLO

model = YOLO("my_model.pt")
cap = cv2.VideoCapture(0)

cap.set(3, 1280) 
cap.set(4, 720)   

class_names = [
    "Arduino Uno",
    "DHT11",
    "ESP32",
    "HC-SR04",
    "LCD 16x2 with I2C",
    "Soil Moisture Sensor"
]

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from camera")
        break

    results = model(frame, stream=True)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{class_names[cls]} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("YOLO Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
