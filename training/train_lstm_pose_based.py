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
sequence_length = 6
num_landmarks = 33
input_size = num_landmarks * 3
hidden_size = 64
epochs = 50
lr = 0.001
foot_index = 32
ball_index = 100

def load_pose_impact_data(base_dir='keypoints_csv_with_ball'):
    X, y = [], []
    for label in os.listdir(base_dir):
        folder = os.path.join(base_dir, label)
        if not os.path.isdir(folder): continue
        for file_name in os.listdir(folder):
            if not file_name.endswith(".npy"): continue
            path = os.path.join(folder, file_name)
            try:
                data = np.load(path)
            except:
                continue
            if data.ndim != 3 or data.shape[1] <= ball_index or data.shape[1] < num_landmarks:
                continue
            foot = data[:, foot_index, :3]
            ball = data[:, ball_index, :3]
            if np.all(ball == 0): continue
            valid = np.where(~np.all(ball == 0, axis=1))[0]
            if len(valid) == 0: continue
            distances = np.linalg.norm(foot[valid] - ball[valid], axis=1)
            impact = valid[np.argmin(distances)]
            start = max(0, impact - 2)
            end = start + sequence_length
            if end > data.shape[0]: continue
            clip = data[start:end, :num_landmarks, :3].reshape(sequence_length, -1)
            X.append(clip)
            y.append(label)
    return np.array(X), np.array(y)

class ImpactLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

def train():
    X, y = load_pose_impact_data()
    if len(X) == 0:
        return

    label_set = sorted(list(set(y)))
    label_to_idx = {lbl: i for i, lbl in enumerate(label_set)}
    idx_to_label = {i: lbl for lbl, i in label_to_idx.items()}
    y_idx = np.array([label_to_idx[v] for v in y])
    num_classes = len(label_to_idx)

    min_samples_per_class = 2
    class_counts = Counter(y_idx)
    valid_indices = [i for i, label_idx_val in enumerate(y_idx) if class_counts[label_idx_val] >= min_samples_per_class]
    if len(valid_indices) < len(X):
        X = X[valid_indices]
        y_idx = y_idx[valid_indices]
        label_set = sorted(list(set([idx_to_label[i] for i in y_idx])))
        label_to_idx = {lbl: i for i, lbl in enumerate(label_set)}
        idx_to_label = {i: lbl for lbl, i in label_to_idx.items()}
        y_idx = np.array([label_to_idx[idx_to_label[orig_idx]] for orig_idx in y_idx])
        num_classes = len(label_to_idx)

    if len(X) == 0 or num_classes == 0:
        return

    test_set_size = 0.4
    if len(X) * (1 - test_set_size) < num_classes or len(X) * test_set_size < num_classes:
        if len(X) > num_classes * min_samples_per_class * 2:
            test_set_size = 0.2
        else:
            return

    X_train, X_test, y_train, y_test = train_test_split(X, y_idx, test_size=test_set_size, stratify=y_idx, random_state=42)

    device = torch.device("cpu")
    model = ImpactLSTM(input_size, hidden_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

    losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # 손실 그래프 저장
    param_str = f"h{hidden_size}_lm{num_landmarks}_fp32"
    loss_fn = f"impact_lstm_loss_{param_str}.png"
    plt.figure()
    plt.plot(losses)
    plt.title(f"Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(loss_fn)
    plt.close()

    # 모델 평가
    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    with torch.no_grad():
        pred_fp32 = model(X_test_tensor).argmax(dim=1)

    unique_lbls_fp32 = sorted(list(set(y_test_tensor.cpu().numpy()) | set(pred_fp32.cpu().numpy())))
    target_names_fp32 = [idx_to_label.get(i, f"Class_{i}") for i in unique_lbls_fp32]
    print(classification_report(y_test_tensor.cpu().numpy(), pred_fp32.cpu().numpy(),
                                target_names=target_names_fp32, labels=unique_lbls_fp32, zero_division=0))

    # 동적 양자화
    quantized_model = torch.quantization.quantize_dynamic(model.to("cpu"),
                                                          {torch.nn.LSTM, torch.nn.Linear},
                                                          dtype=torch.qint8)

    # 모델 저장
    model_dir = "models/weights"
    os.makedirs(model_dir, exist_ok=True)
    quantized_full_model_path = os.path.join(model_dir, f'impact_model_quantized_dynamic_h{hidden_size}.pth')
    torch.save(quantized_model, quantized_full_model_path)

if __name__ == '__main__':
    train()
