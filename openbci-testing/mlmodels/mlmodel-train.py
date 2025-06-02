import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# 1) Load data
df = pd.read_csv('filtered_data.csv')

# Drop timestamp
df = df.drop(columns=['timestamp'])

# Encode mode labels
le = LabelEncoder()
df['mode_label'] = le.fit_transform(df['mode'])

# Features and labels
features = df.drop(columns=['mode', 'mode_label']).values  # shape (N, 11)
labels = df['mode_label'].values

# Scale features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# 2) Create sliding windows
WINDOW_SIZE = 250  # e.g., 1 second at 250Hz
STEP_SIZE = 125    # 50% overlap

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

model = EEG_LSTM(input_size, hidden_size, num_layers, num_classes)

# 6) Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    return correct / total

EPOCHS = 20
for epoch in range(1, EPOCHS + 1):
    train_loss = train(model, train_loader)
    test_acc = evaluate(model, test_loader)
    print(f"Epoch {epoch}: Train loss {train_loss:.4f} | Test accuracy {test_acc:.4f}")
