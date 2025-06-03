import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class EEGNet(nn.Module):
    def __init__(self, num_channels=8, num_classes=4, samples=250):
        super(EEGNet, self).__init__()

        # First block: Temporal convolution
        self.temporal_conv = nn.Conv2d(1, 8, kernel_size=(1, 64), padding=(0, 32), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(8)

        # Second block: Depthwise convolution
        self.depthwise_conv = nn.Conv2d(8, 16, kernel_size=(num_channels, 1), groups=8, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(0.1)

        # Third block: Separable convolution
        self.separable_conv1 = nn.Conv2d(16, 16, kernel_size=(1, 16), padding=(0, 8), bias=False)
        self.separable_conv2 = nn.Conv2d(16, 16, kernel_size=(1, 1), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(16)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout2 = nn.Dropout(0.1)

        # Compute output shape after convolutions to define final Dense layer
        self._output_size = self._get_conv_output_size(num_channels, samples)
        self.classifier = nn.Linear(self._output_size, num_classes)

    def _get_conv_output_size(self, channels, samples):
        with torch.no_grad():
            x = torch.zeros(1, 1, channels, samples)
            x = self.temporal_conv(x)
            x = self.batchnorm1(x)
            x = self.depthwise_conv(x)
            x = self.batchnorm2(x)
            x = self.elu1(x)
            x = self.pool1(x)
            x = self.dropout1(x)
            x = self.separable_conv1(x)
            x = self.separable_conv2(x)
            x = self.batchnorm3(x)
            x = self.elu2(x)
            x = self.pool2(x)
            x = self.dropout2(x)
            return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.batchnorm1(x)
        x = self.depthwise_conv(x)
        x = self.batchnorm2(x)
        x = self.elu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.separable_conv1(x)
        x = self.separable_conv2(x)
        x = self.batchnorm3(x)
        x = self.elu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        x = self.X[idx]
        x = x.T.unsqueeze(0)    # shape: (1, 8, 250)
        return x, self.y[idx]


if __name__ == "__main__":

    # 1) Load data
    data_path = 'Data/bci_log_06030158.csv'
    df = pd.read_csv(data_path)
    df = df.drop(columns=['timestamp'])

    # Features and labels
    # featurenames = ['ch_1','ch_2','ch_3','ch_4',
    #                 'ch_5','ch_6','ch_7','ch_8',
    #                 'accel_0','accel_1','accel_2']
    # featurenames_no_accel = ['ch_1','ch_2','ch_3','ch_4',
    #                         'ch_5','ch_6','ch_7','ch_8']

    featurenames = ['ch_1_mu', 'ch_2_mu', 'ch_3_mu', 'ch_4_mu', 
                    'ch_5_mu', 'ch_6_mu', 'ch_7_mu', 'ch_8_mu', 
                    'ch_1_beta', 'ch_2_beta', 'ch_3_beta', 'ch_4_beta', 
                    'ch_5_beta', 'ch_6_beta', 'ch_7_beta', 'ch_8_beta', 
                    'accel_0', 'accel_1', 'accel_2']
    featurenames_no_accel = ['ch_1_mu', 'ch_2_mu', 'ch_3_mu', 'ch_4_mu', 
                    'ch_5_mu', 'ch_6_mu', 'ch_7_mu', 'ch_8_mu', 
                    'ch_1_beta', 'ch_2_beta', 'ch_3_beta', 'ch_4_beta', 
                    'ch_5_beta', 'ch_6_beta', 'ch_7_beta', 'ch_8_beta']
    features = df[featurenames_no_accel].values

    le = LabelEncoder()
    df['mode_label'] = le.fit_transform(df['mode'])
    labels = df['mode_label'].values

    # Scale features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # 2) Create sliding windows
    WINDOW_SIZE = 250  # e.g., 1 second at 250Hz
    STEP_SIZE = 20    # 50% overlap

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

    # Training and testing data
    X_train, y_train = create_windows(X_train_seq, y_train_seq, WINDOW_SIZE, STEP_SIZE)
    X_test, y_test = create_windows(X_test_seq, y_test_seq, WINDOW_SIZE, STEP_SIZE)

    train_dataset = EEGDataset(X_train, y_train)
    test_dataset = EEGDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    input_size = X_train.shape[2]  # 11 features
    hidden_size = 128
    num_layers = 2
    num_classes = len(le.classes_)
    print(f"{num_classes=}")

    model = EEGNet(num_channels=16, num_classes=3, samples=250)
    # model = EEGNet(input_size, hidden_size, num_layers, num_classes)

    # 6) Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

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
    save_path = 'Models/eegnet_best2.pth'
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
