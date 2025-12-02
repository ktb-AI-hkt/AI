import os
import torch
from model import LeNet

# 1) 파일 존재 여부 확인
path = "./saved_model.pth"

if os.path.exists(path):
    print("✓ saved_model.pth 파일이 존재합니다!")
else:
    print("✗ saved_model.pth 파일이 없습니다...")
    exit()

# 2) 모델 로드 테스트
try:
    model = LeNet()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    print("✓ saved_model.pth 를 정상적으로 로드했습니다!")
except Exception as e:
    print("✗ 모델 로드 실패!", e)
