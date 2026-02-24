import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from collections import Counter
import torch.quantization

# 설정값
sequence_length = 11
num_landmarks = 101
input_size = num_landmarks * 3 + 2
hidden_size = 64
epochs = 50
lr = 0.001
right_foot_index = 32
ball_index = 100

def load_foot_data(base_dir='keypoints_foot_with_ball'):
    X, y = [], []
    for label_name in os.listdir(base_dir):
        folder = os.path.join(base_dir, label_name)
        if not os.path.isdir(folder): continue
        for file_name in os.listdir(folder):
            if not file_name.endswith('.npy'): continue
            path = os.path.join(folder, file_name)
            data = np.load(path)
            if data.ndim != 3 or data.shape[1] <= ball_index or data.shape[1] < num_landmarks:
                continue
            right_foot = data[:, right_foot_index, :3]
            ball = data[:, ball_index, :3]
            if np.all(ball == 0):
                continue
            valid_indices = np.where(~np.all(ball == 0, axis=1))[0]
            if len(valid_indices) == 0:
                continue
            distances = np.linalg.norm(right_foot[valid_indices] - ball[valid_indices], axis=1)
            impact = valid_indices[np.argmin(distances)]
            start = impact - 5
            end = impact + 6
            if start < 0 or end > data.shape[0] or end - start != sequence_length:
                continue
            pose_clip = data[start:end, :num_landmarks, :3].reshape(sequence_length, -1)
            impact_distance = np.linalg.norm(right_foot[impact] - ball[impact])
            dy = right_foot[impact][1] - ball[impact][1]
            additional_features = np.tile([impact_distance, dy], (sequence_length, 1))
            full_clip = np.concatenate([pose_clip, additional_features], axis=1)
            if full_clip.shape[1] != input_size:
                continue
            X.append(full_clip)
            y.append(label_name)
    return np.array(X), np.array(y)

class FootLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

def train():
    X, y = load_foot_data()
    if len(X) == 0:
        print("데이터 없음. 학습 중단.")
        return

    label_set = sorted(list(set(y)))
    label_to_idx = {lbl: i for i, lbl in enumerate(label_set)}
    idx_to_label = {i: lbl for lbl, i in label_to_idx.items()}
    y_idx = np.array([label_to_idx[lbl] for lbl in y])
    num_classes = len(label_to_idx)

    # 최소 샘플 수 미만 클래스 제거
    min_class_count = 2
    filtered_indices = [i for i, lbl_idx in enumerate(y_idx) if Counter(y_idx)[lbl_idx] >= min_class_count]
    if len(filtered_indices) < len(y_idx):
        X = X[filtered_indices]
        y_idx = y_idx[filtered_indices]
        label_set = sorted(list(set([idx_to_label[i] for i in y_idx])))
        label_to_idx = {lbl: i for i, lbl in enumerate(label_set)}
        idx_to_label = {i: lbl for lbl, i in label_to_idx.items()}
        y_idx = np.array([label_to_idx[idx_to_label[orig_idx]] for orig_idx in y_idx])
        num_classes = len(label_to_idx)

    if len(X) == 0 or num_classes == 0:
        print("유효한 학습 데이터 또는 클래스 없음.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y_idx, test_size=0.4, stratify=y_idx, random_state=42)

    device = torch.device("cpu")
    model = FootLSTM(input_size, hidden_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)

    losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")

    # 손실 그래프 저장
    param_str = f"h{hidden_size}_lm{num_landmarks}_fp32"
    loss_fn = f"foot_lstm_loss_{param_str}.png"
    plt.figure()
    plt.plot(losses)
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(loss_fn)
    plt.close()

    # 모델 평가
    model.eval()
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)
    with torch.no_grad():
        pred_fp32 = model(X_test_t).argmax(dim=1)

    unique_labels = sorted(set(y_test_t.cpu().numpy()) | set(pred_fp32.cpu().numpy()))
    target_names = [idx_to_label.get(i, f"Class_{i}") for i in unique_labels]
    print(classification_report(y_test_t.cpu().numpy(), pred_fp32.cpu().numpy(),
                                target_names=target_names, labels=unique_labels, zero_division=0))

    # 모델 동적 양자화
    quantized_model = torch.quantization.quantize_dynamic(
        model.to("cpu"), {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8
    )

    # 모델 저장
    model_dir = "models/weights"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'foot_lstm_model_quantized_dynamic_h{hidden_size}.pth')
    torch.save(quantized_model, model_path)

if __name__ == '__main__':
    train()
