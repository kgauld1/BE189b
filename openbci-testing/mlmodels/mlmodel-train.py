import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# 1) Load data
df = pd.read_csv('filtered_528_nodup.csv')

# Drop timestamp
df = df.drop(columns=['timestamp'])

# Encode mode labels
le = LabelEncoder()
df['mode_label'] = le.fit_transform(df['mode'])

# Features and labels
features = df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0','mode', 'mode_label', 'filtered_mode']).values
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
        label = np.bincount(y_window).argmax()
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
# class EEG_LSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_classes):
#         super(EEG_LSTM, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, num_classes)
#     def forward(self, x):
#         # x shape: (batch_size, seq_len, input_size)
#         out, _ = self.lstm(x)  # out: (batch_size, seq_len, hidden_size)
#         out = out[:, -1, :]    # take output at last time step
#         out = self.fc(out)
#         return out
class EEG_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super(EEG_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

input_size = X_train.shape[2]  # 11 features
hidden_size = 128
num_layers = 2
num_classes = len(le.classes_)
print(f"{num_classes=}")

model = EEG_LSTM(input_size, hidden_size, num_layers, num_classes)

# 6) Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# 7) Training loop
def train(model, loader):
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

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()*X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    return correct / total, total_loss/len(loader.dataset)

EPOCHS = 20
best_val_loss = np.inf
save_path = 'lstm_best.pth'
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
