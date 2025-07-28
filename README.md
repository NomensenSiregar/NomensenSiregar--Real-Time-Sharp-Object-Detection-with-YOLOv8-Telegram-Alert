🔪 Real-Time Sharp Object Detection with YOLOv8 + Telegram Alert

This project implements a real-time object detection system using YOLOv8, specifically trained to detect sharp objects such as knives and scissors. It combines a custom-trained model with the COCO pre-trained YOLOv8 to also recognize general objects like people.

When a sharp object is detected, the system automatically:

Captures the frame, and

Sends an alert message with the image to a Telegram bot in real-time.

🔧 Features:
Real-time detection from webcam or video source

Dual-model architecture: COCO + Custom objects

Telegram bot integration for automatic threat alerting

Time-based throttling to prevent spam alerts

Easily customizable and extendable for other object classes
