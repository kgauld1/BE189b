from braindecode.models import EEGNetv4
import torch
import warnings

warnings.filterwarnings("ignore")

in_chans = 11
window_samp = 250
n_classes = 5

X_torch = torch.randn(size=(50, in_chans, window_samp))  # size: (batch, in_chans, input_window_samples)
y_torch = torch.randint(low=0, high=n_classes, size=(50,))  # size: (batch), values: 0 or 1

module = EEGNetv4(in_chans=in_chans, n_classes=n_classes, input_window_samples=window_samp)

y_pred = module(X_torch)
print('y_pred.shape =', y_pred.shape)  # size: (batch, n_classes)