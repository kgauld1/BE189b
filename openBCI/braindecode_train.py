from braindecode.models import EEGNetv1
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import os

# Dataset class
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

LABEL_MAP = {
    'close': 0,
    'pinch': 1,
    'rest': 2,
    'close-ARM': 0,
    'pinch-LEG': 1,
    0: 'close',
    1: 'pinch',
    2: 'rest'
}

# 1) Load data
data_path = 'Data/bci_log_withemg.csv'
df = pd.read_csv(data_path)
df = df.drop(columns=['timestamp'])
df['mode_label'] = [LABEL_MAP[k] for k in df['mode']]

featurenames = [
    'ch_1_mu', 'ch_2_mu', 'ch_3_mu', 'ch_4_mu', 
    'ch_5_mu', 'ch_6_mu', 'ch_7_mu', #'ch_8_mu', 
    'ch_1_beta', 'ch_2_beta', 'ch_3_beta', 'ch_4_beta', 
    'ch_5_beta', 'ch_6_beta', 'ch_7_beta', 'ch_8_emg'
]

features = df[featurenames].values
labels = df['mode_label'].values

# Scale features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# 2) Create sliding windows
WINDOW_SIZE = 250
STEP_SIZE = 100
split_ratio = 0.8
split_index = int(len(features) * split_ratio)

X_train_seq, y_train_seq = features[:split_index], labels[:split_index]
X_test_seq, y_test_seq = features[split_index:], labels[split_index:]

def create_windows(features, labels, window_size, step_size):
    X, y = [], []
    for start in range(0, len(features) - window_size + 1, step_size):
        end = start + window_size
        X_window = features[start:end]
        y_window = labels[start:end]
        label = np.bincount(y_window).argmax()
        X.append(X_window)
        y.append(label)
    return np.array(X), np.array(y)

X_train, y_train = create_windows(X_train_seq, y_train_seq, WINDOW_SIZE, STEP_SIZE)
X_test, y_test = create_windows(X_test_seq, y_test_seq, WINDOW_SIZE, STEP_SIZE)

# 3) Dataset and DataLoader
train_dataset = EEGDataset(X_train, y_train)
test_dataset = EEGDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# 4) DeepConvNet (Deep4Net) model
input_size = X_train.shape[2]
num_classes = len(np.unique(y_train))

model = EEGNetv1(
    n_chans=15,
    n_outputs=3,
    n_times=WINDOW_SIZE,
    final_conv_length='auto'
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 5) Training and evaluation
def train(model, loader):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        X_batch = X_batch.permute(0, 2, 1)  # (batch, channels, time)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            X_batch = X_batch.permute(0, 2, 1)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    print(classification_report(all_labels, all_preds))
    return correct / total, total_loss / len(loader.dataset)

# 6) Training loop
EPOCHS = 50
best_val_loss = np.inf
save_path = 'Models/braindecode_best_emg.pth'
os.makedirs('Models', exist_ok=True)

for epoch in range(1, EPOCHS + 1):
    train_loss = train(model, train_loader)
    val_acc, val_loss = evaluate(model, test_loader)
    print(f"Epoch {epoch}: Train loss {train_loss:.4f} | Test accuracy {val_acc:.4f} | Val loss {val_loss:.4f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
        }, save_path)
        print(f"Model saved at epoch {epoch} with val loss {val_loss:.4f}")
