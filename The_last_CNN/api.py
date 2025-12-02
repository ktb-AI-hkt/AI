import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import numpy as np

from model import LeNet

app = FastAPI()

# -------------------
# 1. 모델 로드
# -------------------
device = torch.device("cpu")
model = LeNet().to(device)
model.load_state_dict(torch.load("./saved_model.pth", map_location=device))
model.eval()

# -------------------
# 2. 이미지 → Tensor 변환 함수
# -------------------
def preprocess_image(img: Image.Image):
    img = img.convert("L")              # grayscale
    img = img.resize((28, 28))          # MNIST 크기
    img = np.array(img).astype("float32") / 255.0
    img = (img - 0.5) / 0.5             # normalize
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0)
    return img

# -------------------
# 3. Predict 엔드포인트
# -------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # 이미지 읽기
    img = Image.open(file.file)
    x = preprocess_image(img)
    x = x.to(device)

    # 모델 예측
    with torch.no_grad():
        outputs = model(x)
        predicted = outputs.argmax(dim=1).item()

    return {"prediction": predicted}