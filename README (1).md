# 🔪 Deteksi Benda Tajam Real-Time dengan YOLOv8 dan Notifikasi Telegram

Proyek ini mengimplementasikan sistem **deteksi objek real-time** menggunakan **YOLOv8** yang telah dilatih secara khusus untuk mendeteksi benda tajam seperti **pisau** dan **gunting**. Sistem ini juga menggabungkan model YOLOv8 bawaan (pre-trained COCO) untuk mendeteksi objek umum seperti **manusia**, sehingga dapat memberikan konteks visual terhadap deteksi yang berbahaya.

## 🚀 Fitur Utama

- ✅ Deteksi **real-time** menggunakan webcam atau video
- 🔍 Penggabungan **dua model YOLOv8** (COCO dan custom)
- 📸 Tangkapan layar otomatis saat objek tajam terdeteksi
- 📩 Kirim **notifikasi ke Telegram** secara otomatis melalui bot
- ⏱️ Fitur pengaturan delay alert agar tidak spam
- 🔧 Kode mudah disesuaikan untuk objek lain

## 🗂️ Struktur Dataset

Dataset custom menggunakan struktur standar YOLO:

```
dataset_ml/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── data.yaml
```

Pastikan `data.yaml` memiliki format seperti ini:

```yaml
train: ../dataset_ml/train/images
val: ../dataset_ml/valid/images

nc: 2
names: ['knife', 'scissors']
```

## 🛠️ Instalasi

1. Clone repository:
   ```bash
   git clone https://github.com/username/deteksi-benda-tajam-yolov8.git
   cd deteksi-benda-tajam-yolov8
   ```

2. Install dependencies:
   ```bash
   pip install ultralytics opencv-python python-telegram-bot
   ```

3. (Opsional) Buat virtual environment jika diinginkan.

## 📦 Pelatihan Model

Untuk melatih model YOLOv8 dengan dataset custom:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # gunakan model base
model.train(
    data="dataset_ml/data.yaml",
    epochs=50,
    imgsz=640,
    name="knife_scissors_custom"
)
```

Model hasil pelatihan (`best.pt`) akan disimpan di:
```
runs/detect/knife_scissors_custom/weights/best.pt
```

## 🎥 Deteksi Real-Time & Alert Telegram

Script utama:

```python
from ultralytics import YOLO
import cv2
import time
import telegram

# Inisialisasi model
model1 = YOLO("yolov8n.pt")  # model COCO
model2 = YOLO("runs/detect/knife_scissors_custom/weights/best.pt")  # model custom

# Inisialisasi Telegram
bot_token = "YOUR_BOT_TOKEN"
chat_id = "YOUR_CHAT_ID"
bot = telegram.Bot(token=bot_token)

last_alert_time = 0
alert_delay = 10  # detik

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results1 = model1.predict(source=frame, conf=0.5, verbose=False)
    results2 = model2.predict(source=frame, conf=0.5, verbose=False)

    combined = results1[0].plot()
    combined = results2[0].plot()

    cv2.imshow("Deteksi Benda Tajam", combined)

    names = results2[0].names
    classes = results2[0].boxes.cls.tolist()
    current_time = time.time()

    if classes and (current_time - last_alert_time > alert_delay):
        cv2.imwrite("deteksi.jpg", frame)
        with open("deteksi.jpg", "rb") as f:
            bot.send_photo(chat_id=chat_id, photo=f, caption="⚠️ Benda tajam terdeteksi!")
        last_alert_time = current_time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## ✅ Contoh Output

- Deteksi **manusia + benda tajam**
- Kirim foto otomatis ke Telegram saat deteksi
- Antarmuka real-time dengan anotasi bounding box

## 🤖 Integrasi Telegram

1. Buat bot: cari `@BotFather` di Telegram
2. Dapatkan **token**
3. Cari `@userinfobot` untuk tahu **chat ID**
4. Ganti `YOUR_BOT_TOKEN` dan `YOUR_CHAT_ID` di script

## 📌 Catatan

- Pastikan path dataset dan model sesuai
- Gunakan GPU untuk performa lebih baik jika tersedia
- Telegram alert diatur delay agar tidak spam

## 📄 Lisensi

MIT License. Bebas digunakan untuk riset, pembelajaran, atau pengembangan pribadi.

## 📬 Kontak

Jika ada pertanyaan, silakan hubungi saya atau buat [issue](https://github.com/username/deteksi-benda-tajam-yolov8/issues) di repositori ini.
