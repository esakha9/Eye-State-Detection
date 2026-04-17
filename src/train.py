# =========================
# TRAINING SCRIPT
# =========================

import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.utils.data import DataLoader

# =========================
# Paths
# =========================
train_dir = "dataset/train"
val_dir = "dataset/val"

# =========================
# Transforms
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# Load Data
# =========================
train_data = datasets.ImageFolder(train_dir, transform=transform)
val_data = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

print("Classes:", train_data.classes)

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Model
# =========================
model = models.resnet18(weights="DEFAULT")
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# =========================
# Loss & Optimizer
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# =========================
# Training Loop
# =========================
epochs = 5

for epoch in range(epochs):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# =========================
# Save Model
# =========================
torch.save(model.state_dict(), "eye_classifier.pth")

print("Model training complete and saved!")