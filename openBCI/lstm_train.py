import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

import torch.nn.functional as F


# 1) Load data
data_path = 'Data/bci_log_withemg.csv'
df = pd.read_csv(data_path)

# Drop timestamp
df = df.drop(columns=['timestamp'])

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
# Encode mode labels
df['mode_label'] = [LABEL_MAP[k] for k in df['mode']]

# Features and labels
# features = df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0','mode', 'mode_label', 'filtered_mode']).values
featurenames = ['ch_1_mu', 'ch_2_mu', 'ch_3_mu', 'ch_4_mu', 
                'ch_5_mu', 'ch_6_mu', #'ch_7_mu', #'ch_8_mu', 
                'ch_1_beta', 'ch_2_beta', 'ch_3_beta', 'ch_4_beta', 
                'ch_5_beta', 'ch_6_beta', 'ch_7_emg', 'ch_8_emg']
                #'accel_0', 'accel_1', 'accel_2']
# featurenames = ['ch_1','ch_2','ch_3','ch_4',
#                 'ch_5','ch_6','ch_7','ch_8',
#                 'accel_0','accel_1','accel_2']
# featurenames = ['ch_8_emg']
features = df[featurenames].values
labels = df['mode_label'].values


# Scale features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# 2) Create sliding windows
WINDOW_SIZE = 250  # e.g., 1 second at 250Hz
STEP_SIZE = 75    # 50% overlap

split_ratio = 0.8
split_index = int(len(features) * split_ratio)

# Split BEFORE windowing:
X_train_seq = features[:split_index]
y_train_seq = labels[:split_index]

X_test_seq = features[split_index:]
y_test_seq = labels[split_index:]

# Then window each separately with same function:
def create_windows(features, labels, window_size, step_size):
    X = []
    y = []
    for start in range(0, len(features) - window_size + 1, step_size):
        end = start + window_size
        X_window = features[start:end]
        y_window = labels[start:end]
        # label = np.bincount(y_window).argmax()
        label = labels[start + window_size // 2]
        X.append(X_window)
        y.append(label)
    return np.array(X), np.array(y)


X_train, y_train = create_windows(X_train_seq, y_train_seq, WINDOW_SIZE, STEP_SIZE)
X_test, y_test = create_windows(X_test_seq, y_test_seq, WINDOW_SIZE, STEP_SIZE)


# 4) Dataset class
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = EEGDataset(X_train, y_train)
test_dataset = EEGDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# 5) Define LSTM model

class EEG_LSTM(nn.Module):
    def __init__(self, input_size=15, hidden_size=128, num_layers=2, num_classes=3, dropout_prob=0.3):
        super(EEG_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout_prob,
                            bidirectional=True)  # Better feature learning

        self.dropout = nn.Dropout(dropout_prob)

        # Hidden size doubled due to bidirectionality
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)  # out: (batch_size, seq_length, hidden_size*2)
        # out = out[:, -1, :]    # Take the output of the last time step
        out = torch.mean(out, dim=1)  # (batch_size, hidden_size * 2)
        out = self.dropout(out)
        out = self.fc(out)
        return out

# class DualStreamLSTM(nn.Module):
#     def __init__(self, eeg_channels, emg_channels, hidden_size, num_classes):
#         super().__init__()
#         self.eeg_lstm = nn.LSTM(eeg_channels, hidden_size, batch_first=True, bidirectional=True)
#         self.emg_lstm = nn.LSTM(emg_channels, hidden_size, batch_first=True, bidirectional=True)
#         self.dropout = nn.Dropout(0.3)
#         self.fc = nn.Linear(2 * hidden_size * 2, num_classes)  # 2 for biLSTM x2 streams

#     def forward(self, x):
#         eeg = x[:, :, :14]  # channels 0-13 = EEG
#         emg = x[:, :, 14:]  # channel 14 = EMG

#         eeg_out, _ = self.eeg_lstm(eeg)
#         emg_out, _ = self.emg_lstm(emg)

#         eeg_feat = eeg_out.mean(dim=1)
#         emg_feat = emg_out.mean(dim=1)

#         combined = torch.cat([eeg_feat, emg_feat], dim=1)
#         return self.fc(self.dropout(combined))


input_size = X_train.shape[2]  # 11 features
hidden_size = 128
num_layers = 2
num_classes = 3
print(f"{num_classes=}")

print(f"{input_size=}, {hidden_size=}, {num_layers=}, {num_classes=}")

model = EEG_LSTM(input_size, hidden_size, num_layers, num_classes)

# 6) Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = model.to(device)


class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights[0] *= 1.1
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        return loss.mean() if self.reduction == 'mean' else loss.sum()

# criterion = nn.CrossEntropyLoss(weight=class_weights)
criterion = FocalLoss(alpha=1.0, gamma=2.0, reduction='mean', weight=class_weights)
# criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 7) Training loop
def train(model, loader):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        # print(f"X_batch shape: {X_batch.shape}, y_batch shape: {y_batch.shape}")
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
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()*X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    print(classification_report(all_labels, all_preds))
    return correct / total, total_loss/len(loader.dataset), f1_score(all_labels, all_preds, average='macro')

EPOCHS = 50
best_f1 = 0
save_path = 'Models/lstm_best_merged.pth'
backup_save_path = 'Models/lstm_best_merged_backup.pth'
for epoch in range(1, EPOCHS + 1):
    train_loss = train(model, train_loader)
    val_acc, val_loss, f1 = evaluate(model, test_loader)
    print(f"Epoch {epoch}: Train loss {train_loss:.4f} | Test accuracy {val_acc:.4f} | Val loss {val_loss:.4f}")
    if f1 > best_f1:
        best_f1 = f1
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
        }, save_path)
        print(f"Model saved at epoch {epoch} with f1 {best_f1:.4f}")
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
        }, backup_save_path)