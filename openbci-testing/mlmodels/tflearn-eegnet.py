import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
from huggingface_hub import hf_hub_download

# === Data Preprocessing ===
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

def load_and_preprocess(csv_path, window_size=250, step_size=125, split_ratio=0.8):
    df = pd.read_csv(csv_path)
    df = df.drop(columns=['timestamp'])
    le = LabelEncoder()
    df['mode_label'] = le.fit_transform(df['mode'])
    features = df.drop(columns=['Unnamed: 0', 'mode', 'mode_label', 'filtered_mode']).values
    labels = df['mode_label'].values

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    split_index = int(len(features) * split_ratio)
    X_train_seq, y_train_seq = features[:split_index], labels[:split_index]
    X_test_seq, y_test_seq = features[split_index:], labels[split_index:]

    X_train, y_train = create_windows(X_train_seq, y_train_seq, window_size, step_size)
    X_test, y_test = create_windows(X_test_seq, y_test_seq, window_size, step_size)

    return X_train, y_train, X_test, y_test, le

# === Dataset Class ===
class EEGNetDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1).permute(0, 1, 3, 2)  # (N, 1, 11, 250)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# === Training Utilities ===
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
    return total_loss / len(loader.dataset), correct / total

# === Config ===
csv_path = 'filtered_data_528.csv'
window_size = 250
step_size = 125
batch_size = 64
epochs = 30
save_path = 'eegnet_tf_model.pth'

# === Load pretrained model ===
path_kwargs = hf_hub_download(repo_id='PierreGtch/EEGNetv4', filename='EEGNetv4_Lee2019_MI/kwargs.pkl')
path_params = hf_hub_download(repo_id='PierreGtch/EEGNetv4', filename='EEGNetv4_Lee2019_MI/model-params.pkl')

with open(path_kwargs, 'rb') as f:
    kwargs = pickle.load(f)

# Update for 11-channel input and 5 output classes
module_kwargs = kwargs['module_kwargs']
module_kwargs['n_chans'] = 11
module_kwargs['n_times'] = 250
module_kwargs['n_outputs'] = 5
module_kwargs.pop('in_chans', None)
module_kwargs.pop('n_classes', None)
module_kwargs.pop('input_window_samples', None)

module_cls = kwargs['module_cls']
model = module_cls(**module_kwargs)

# Load compatible weights only
pretrained_weights = torch.load(path_params, map_location='cpu')
model_dict = model.state_dict()

compatible_weights = {
    k: v for k, v in pretrained_weights.items()
    if k in model_dict and v.size() == model_dict[k].size()
}

model_dict.update(compatible_weights)
model.load_state_dict(model_dict)

# Move model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze final classifier
for param in model.final_layer.conv_classifier.parameters():
    param.requires_grad = True

# === Load Data ===
X_train, y_train, X_test, y_test, label_encoder = load_and_preprocess(csv_path, window_size, step_size)
train_dataset = EEGNetDataset(X_train, y_train)
test_dataset = EEGNetDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# === Optimizer & Loss ===
optimizer = torch.optim.Adam(model.final_layer.conv_classifier.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# === Training Loop ===
best_val_loss = float('inf')
for epoch in range(1, epochs + 1):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = eval_epoch(model, test_loader, criterion, device)

    print(f"Epoch {epoch}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f}")

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
