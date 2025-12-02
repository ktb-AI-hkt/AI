import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

# nn.conv2d(in_channels,out_channels,kernel_size,stride)
# nn.MaxPool(filter_size,stride)
# stride=1, padding=0
# 모델 수정 바람
# 가상 환경: real_resnet
# 데이터셋: 뭐..쓰지..? -> cifar-10 아님 mnist

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
 # 이럴거면 댱 resnet을 쓰지...
 # 또 고민의 순간이 왔따... nn.Sequential을 쓸 것인가 말것이가 // 일단 써보자
        self.layer1=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=6,kernel_size=3,padding=1,stride=1), # kernel_size는 전체적인 구조를 볼 필요는 없을 것 같아 3x3으로 줄인다.
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2) # output_size = 14x14
            #nn.Tanh(), # activation function, tanh -> relu
            #nn.AvgPool2d(kernel_size=2,stride=2)
        )
        self.layer2=nn.Sequential(
            nn.Conv2d(in_channels=6,out_channels=16,kernel_size=3,padding=1,stride=1), # intput__size = 14x14
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)  # output_size = output_size=7x7
            #nn.Tanh(), # activation function tanh -> relu
            #nn.AvgPool2d(kernel_size=2,stride=2)
        )
        # self.layer3=nn.Sequential(
        #     nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2,stride=2)
        # ) # 이런식으로 바꿔봐야 겠다.
        # self.layer4 = nn.Sequential(
        #     nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3),
        #     nn.ReLU()
        # )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding=1,stride=1), # input_size=7x7
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2,stride=2) # output_size=7x7x32
        )

        self.fc1=nn.Linear(7*7*32,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10) # 왜 10이지? class가 10개인가? yes~ mnist

    def forward(self,x):
        x=self.layer1(x)
        #print("After layer1:", x.shape) # 디버깅용 출력문
        x=self.layer2(x)
        #print("After layer2:", x.shape)
        x=self.layer3(x)
        #print("After layer3:", x.shape)
        x=x.view(x.size(0),-1) # flatten / 수정한 부분
        #print("Flatten size:", x.shape)
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)

        return x   
    
# 모델 코드 지분률: 나:63%, gpt:37% ㅋ

# --------------------------------------------------------------------------
# 문제점: 큰 이미지를 입력했을 때 인식을 잘 하지 못한다.
# 문제 원인 예상: 28x28이라는 작은 이미지로 학습시켜서 큰 이미지에 대해 성능이 좋지 않은 것 같다.
# 문제 분석: 실제 데이터 중 28x28의 작은 이미지를 입력해 본다.
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
# 진짜로 문제가 큰 이미지를 입력했을 때 성능이 안 좋게 나오는게 맞는 것 같다.
# 왜냐면 28x28짜리 이미지를 입력하면 지대로 성능이 나오기 때문이다.
# 문제 해결: 큰 이미지를 학습시킨다. 그리고 큰 이미지에 맞는 모델로 수정을 해본다.
# 방법1. 큰 이미지를 학습시킨다.
# 방법2. layer를 더 깊게 쌓아본다.
#---------------------------------------------------------------------------