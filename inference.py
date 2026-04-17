# =========================
# VIDEO TESTING SCRIPT
# =========================

import torch
import cv2
import pandas as pd
from torchvision import transforms, models
from PIL import Image

# =========================
# Paths
# =========================
MODEL_PATH = "eye_classifier.pth"
VIDEO_PATH = "input.mp4"

OUTPUT_VIDEO = "output_video.mp4"
CSV_PATH = "predictions.csv"

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Load Model
# =========================
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

classes = ['closed', 'open']

# =========================
# Transform
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# Load Video
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(5))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_width, frame_height))

print("Processing video...")

# =========================
# Processing Loop
# =========================
records = []
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Center crop
    h, w, _ = frame.shape
    crop = frame[h//4:h*3//4, w//4:w*3//4]

    img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)

    label = classes[pred.item()]

    # Draw label
    color = (0, 255, 0) if label == 'open' else (0, 0, 255)
    cv2.putText(frame, label, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

    out.write(frame)

    records.append({
        "frame": frame_id,
        "prediction": label
    })

    frame_id += 1

cap.release()
out.release()

print("Video processing complete!")

# =========================
# Save CSV
# =========================
df = pd.DataFrame(records)
df.to_csv(CSV_PATH, index=False)

print("Predictions saved!")