# 🪖 Surveillance Bot for Army

## 📘 Overview
The Surveillance Bot is a hybrid robotics project that combines AI-based object detection and autonomous navigation.  
The system uses a Raspberry Pi for intelligent camera-based threat detection with YOLOv11, and an Arduino for motor and sensor control.  
This integration enables real-time surveillance of restricted or hazardous areas while minimizing risks to soldiers.

---

## 🎯 Objective
To develop an autonomous surveillance robot capable of detecting intruders or weapons using computer vision (YOLOv11) and IoT, ensuring efficient and safe monitoring in defense environments.

---

## ⚙️ Features
- 🔍 Real-time object detection using **YOLOv11**
- 🚗 Autonomous movement using **Arduino** with ultrasonic sensors
- 🧠 Local AI processing on **Raspberry Pi**
- 🌐 IoT-enabled alert and monitoring system
- 🛑 Obstacle avoidance and safe path navigation
- 📡 Serial communication between Pi ↔ Arduino

---

## 🧠 Technologies Used

| Category | Tools / Components |
|-----------|--------------------|
| Hardware| Raspberry Pi 4, Arduino Uno, Pi Camera / USB Webcam, Ultrasonic Sensors, L298N Motor Driver, DC Motors, Chassis, Battery Pack |
| Software | Python, Arduino IDE, YOLOv11, OpenCV, PySerial, MQTT (optional), Raspberry Pi OS |
| Libraries | torch, torchvision, ultralytics, numpy, opencv-python, pyserial, paho-mqtt, flask (optional) |

---

## 🪛 Hardware Setup
1. Raspberry Pi handles the camera and object detection.  
2. Arduino Uno controls motors and ultrasonic sensors for navigation.  
3. Power is supplied using a battery pack or power bank.  

📄 Refer to [`Hardware/circuit_diagram.png`](./Hardware/circuit_diagram.png) for full wiring details.

---

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Add YOLOv11 model weights

Download or train your YOLOv11 model and place it inside:
model/yolov11.pt

4️⃣ Upload Arduino Code


Select the correct COM port and upload it to your Arduino Uno.

5️⃣ Run the Raspberry Pi code
python raspberry_pi/main.py





Surveillance-Bot/
├── raspberry_pi/
│   ├── detection.py
│
├── arduino/
│   ├── surveillance_bot.ino
│   └── pin_config.txt
│
├── model/
│   ├── yolov11.pt
│   ├── texts
│
├── hardware/
│   ├── circuit_diagram.png
│   └── components_list.md
│
├── media/
│   ├── bot_front.jpg
│   ├── bot_side.jpg
│   └── model.jpg
│
├── requirements.txt
└── README.md
