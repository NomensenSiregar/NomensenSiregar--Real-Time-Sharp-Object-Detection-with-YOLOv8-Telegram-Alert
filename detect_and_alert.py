from ultralytics import YOLO
import cv2
import requests
import time

# Telegram Settings
BOT_TOKEN = 'Isi Bot Token anda'
CHAT_ID = 'Isi Chat ID anda' 

# Fungsi kirim alert ke Telegram
def send_alert(image_path, label):
    message = f"🚨 Terdeteksi objek berbahaya: *{label}*"
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    files = {'photo': open(image_path, 'rb')}
    data = {'chat_id': CHAT_ID, 'caption': message, 'parse_mode': 'Markdown'}
    try:
        requests.post(url, files=files, data=data)
    except Exception as e:
        print("Gagal kirim ke Telegram:", e)

# Load kedua model 
model_umum = YOLO("yolov8n.pt")
model_tajam = YOLO("runs/detect/knife_scissors_custom/weights/best.pt")

# Jalankan kamera / video
cap = cv2.VideoCapture(1) 

last_alert_time = 0
alert_interval = 10  # detik

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi objek umum
    result_umum = model_umum(frame, conf=0.5)[0]

    # Deteksi objek tajam
    result_tajam = model_tajam(frame, conf=0.4)[0]

    sharp_detected = False
    label_sharp = ""

    for r in [result_umum, result_tajam]:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = r.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            color = (0, 255, 0)
            if label in ["knife", "scissors"]:
                color = (0, 0, 255)
                sharp_detected = True
                label_sharp = label

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Kirim alert jika benda tajam terdeteksi dan jeda waktu cukup
    current_time = time.time()
    if sharp_detected and current_time - last_alert_time > alert_interval:
        img_path = "alert.jpg"
        cv2.imwrite(img_path, frame)
        send_alert(img_path, label_sharp)
        last_alert_time = current_time

    cv2.imshow("Deteksi Gabungan", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
