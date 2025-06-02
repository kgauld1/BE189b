import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import os

# === EEGNet model definition ===
class EEGNet(nn.Module):
    def __init__(self, num_channels=11, num_samples=250, num_classes=3, dropout=0.5):
        super(EEGNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.depthwiseConv = nn.Conv2d(16, 32, (num_channels, 1), groups=16, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.elu = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)

        self.separableConv = nn.Conv2d(32, 32, (1, 16), padding=(0, 8), groups=32, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.avgpool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)

        self.flattened_size = self._get_flattened_size(num_samples)
        self.classifier = nn.Linear(self.flattened_size, num_classes)

    def _get_flattened_size(self, num_samples):
        size = num_samples
        size = (size + 2*32 - 64 + 1)  # conv1
        size = size // 4               # avgpool1
        size = (size + 2*8 - 16 + 1)   # separable conv
        size = size // 8               # avgpool2
        return 32 * size

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwiseConv(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        x = self.separableConv(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

# === Dataset class ===
class EEGNetDataset(Dataset):
    def __init__(self, X, y):
        # X shape (N, seq_len, features) → (N, 1, features, seq_len)
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# === Windowing function ===
def create_windows(features, labels, window_size, step_size):
    X = []
    y = []
    for start in range(0, len(features) - window_size + 1, step_size):
        end = start + window_size
        X_window = features[start:end]
        y_window = labels[start:end]
        label = np.bincount(y_window).argmax()
        X.append(X_window)
        y.append(label)
    return np.array(X), np.array(y)

# === Load and preprocess your CSV ===
def load_and_preprocess(csv_path, window_size=250, step_size=125, split_ratio=0.8):
    df = pd.read_csv(csv_path)
    df = df.drop(columns=['timestamp'])
    le = LabelEncoder()
    df['mode_label'] = le.fit_transform(df['mode'])

    features = df.drop(columns=['mode', 'mode_label']).values
    labels = df['mode_label'].values

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    split_index = int(len(features) * split_ratio)
    X_train_seq, y_train_seq = features[:split_index], labels[:split_index]
    X_test_seq, y_test_seq = features[split_index:], labels[split_index:]

    X_train, y_train = create_windows(X_train_seq, y_train_seq, window_size, step_size)
    X_test, y_test = create_windows(X_test_seq, y_test_seq, window_size, step_size)

    return X_train, y_train, X_test, y_test, le

# === Training and evaluation functions ===
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy

# === Main training loop ===
def main():
    csv_path = 'bci_log.csv'  # Your CSV path
    window_size = 250
    step_size = 125
    batch_size = 64
    epochs = 30
    save_path = 'best_eegnet_model.pth'

    X_train, y_train, X_test, y_test, label_encoder = load_and_preprocess(
        csv_path, window_size, step_size)

    train_dataset = EEGNetDataset(X_train, y_train)
    test_dataset = EEGNetDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = EEGNet(num_channels=X_train.shape[2], num_samples=X_train.shape[1],
                   num_classes=len(label_encoder.classes_))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(model, test_loader, criterion, device)

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f}")

        # Save model if validation loss decreased
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'label_encoder_classes': label_encoder.classes_
            }, save_path)
            print(f"Model saved at epoch {epoch} with val loss {val_loss:.4f}")

if __name__ == '__main__':
    main()
