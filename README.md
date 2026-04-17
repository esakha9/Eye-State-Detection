# 👁 Eye State Detection (Open / Closed)

## 📌 Overview

This project detects whether a person's eyes are **open or closed** using a deep learning model. It processes video input and predicts eye state frame-by-frame in real time.

---

## 🧠 Model Details

* **Architecture:** ResNet18 (CNN-based classifier)
* **Framework:** PyTorch
* **Task:** Binary Classification (Open vs Closed Eyes)
* **Input:** Face/Eye images
* **Output:** Eye state prediction (Open / Closed)

---

## ⚙️ How it Works

1. Train a deep learning model on labeled eye images (open / closed)
2. Load the trained model (`eye_classifier.pth`)
3. Process video frame-by-frame
4. Predict eye state for each frame
5. Display and save output video with predictions

---

## 📁 Project Files

* `train.py` → Train the model
* `inference.py` → Run detection on video
* `eye_classifier.pth` → Trained model
* `input.mp4` → Input test video
* `output_video.mp4` → Output result video

---

## 🎯 Applications

* Driver drowsiness detection
* Fatigue monitoring systems
* Safety systems in vehicles

---

## 👨‍💻 Author

Esa Khan
AI Engineer
