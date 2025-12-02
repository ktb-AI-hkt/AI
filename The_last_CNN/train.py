# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm   # ✅ 추가
import numpy as np

from model import LeNet

# -------------------------
# 1. Device
# -------------------------
device = torch.device("cpu")

# -------------------------
# 2. Dataset & DataLoader
# -------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# -------------------------
# 3. Model 준비
# -------------------------
model = LeNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------
# 4. Train 함수 (tqdm 적용)
# -------------------------
def train(epoch):
    model.train()
    running_loss = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} Training", leave=False)

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 🔥 tqdm 표시 업데이트
        progress_bar.set_postfix(loss=loss.item())

    return running_loss / len(train_loader)

# -------------------------
# 5. Test 함수
# -------------------------
def test():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100. * correct / total
    print(f"Test Accuracy: {acc:.2f}%")
    return acc

# -------------------------
# 6. 실행 Loop
# -------------------------
for epoch in range(1, 6):
    train_loss = train(epoch)
    print(f"Epoch {epoch} Average Loss: {train_loss:.4f}")
    test()

# 저장
torch.save(model.state_dict(), "./saved_model.pth")