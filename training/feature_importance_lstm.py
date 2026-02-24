import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from train_lstm_pose_based import ImpactLSTM, load_pose_impact_data

label_map = {'front': 0, 'middle': 1, 'back': 2}
sequence_length = 6
num_landmarks = 33
input_size = num_landmarks * 3
hidden_size = 64
ball_index = 100

# BlazePose 관절 이름 (33개)
joint_names = [
    'Nose', 'L Eye Inner', 'L Eye', 'L Eye Outer', 'R Eye Inner', 'R Eye', 'R Eye Outer',
    'L Ear', 'R Ear', 'Mouth L', 'Mouth R', 'L Shoulder', 'R Shoulder',
    'L Elbow', 'R Elbow', 'L Wrist', 'R Wrist',
    'L Pinky', 'R Pinky', 'L Index', 'R Index', 'L Thumb', 'R Thumb',
    'L Hip', 'R Hip', 'L Knee', 'R Knee', 'L Ankle', 'R Ankle',
    'L Heel', 'R Heel', 'L FootIdx', 'R FootIdx'
]

def analyze_importance():
    X, y = load_pose_impact_data()
    if len(X) == 0:
        print("❌ 데이터가 없습니다.")
        return

    print("✅ 전체 클래스 분포:", Counter(y))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImpactLSTM(input_size, hidden_size, num_classes=3).to(device)
    model.load_state_dict(torch.load("models/impact_model.pth"))
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True).to(device)
    y_tensor = torch.tensor(y, dtype=torch.long).to(device)

    criterion = nn.CrossEntropyLoss()
    output = model(X_tensor)
    loss = criterion(output, y_tensor)
    loss.backward()

    grads = X_tensor.grad.abs().detach().cpu().numpy()
    importance = grads.mean(axis=(0, 1))

    joint_importance = []
    for i in range(num_landmarks):
        joint_importance.append(importance[i*3:(i+1)*3].mean())

    # 시각화
    plt.figure(figsize=(18, 6))
    sns.barplot(x=joint_names, y=joint_importance, palette='viridis')
    plt.title("Feature Importance by Joint (Gradient-based)")
    plt.xlabel("Joint Name")
    plt.ylabel("Mean Gradient Magnitude")
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    analyze_importance()
