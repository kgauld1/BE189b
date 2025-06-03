import logging, traceback, csv, threading, time
from datetime import datetime, timezone
from enum import Enum

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations

import pickle
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from eegnet_train import EEGNet

import serial

# class EEG_LSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
#         super(EEG_LSTM, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
#                             batch_first=True, dropout=dropout)
#         self.fc = nn.Sequential(
#             nn.Dropout(dropout),
#             nn.Linear(hidden_size, num_classes)
#         )
#     def forward(self, x):
#         out, _ = self.lstm(x)
#         out = out[:, -1, :]
#         out = self.fc(out)
#         return out

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
    
# from lstm_train import EEG_LSTM

LABEL_MAP = {
    'close': 0,
    'pinch': 1,
    'rest': 2,
    0: 'close',
    1: 'pinch',
    2: 'rest'
}

STATE_MAP = {
    'close': 1,
    'pinch': -1,
    'rest': 0,
    1: 'close',
    -1: 'pinch',
    0: 'rest'
}

class BCI:
    def __init__(self, board_shim, model, arduino):
        self.board_shim = board_shim
        self.arduino = arduino
        self.board_id = board_shim.get_board_id()
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        self.emg_channels = self.exg_channels[-2:]  # last two channels are EMG
        self.eeg_channels = self.exg_channels[:-2]  # all but last two are EEG
        self.accel_channels = BoardShim.get_accel_channels(self.board_id)
        self.timestamp_channel = BoardShim.get_timestamp_channel(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
#        self.window_size = 4
        self.num_points = 250#self.window_size*self.sampling_rate
        self.window_size = self.num_points/self.sampling_rate
        print(f"SETTING WINDOW SIZE TO {self.window_size}")
        self.current_prediction = LABEL_MAP['rest']
        
        self.predictor = model
        
        self.training_mode = False
        self.training_class = LABEL_MAP['rest']

        self.raw_state = 0
        self.thresh_state = 0
        
        self.stopflag = threading.Event()
        self.update_thread = threading.Thread(target=self.update_cycle)
        self.send_thread = threading.Thread(target=self.send_cycle)
        self.update_thread.start()
        self.send_thread.start()
    
    def close(self):
        self.stopflag.set()
        time.sleep(0.1)
        self.update_thread.join()
        self.send_thread.join()
    
    def send_cycle(self):
        try:
            while not self.stopflag.wait(0.01):
                print(f'sending {LABEL_MAP[LABEL_MAP[STATE_MAP[self.thresh_state]]]}')
                self.arduino.write(f"{LABEL_MAP[STATE_MAP[self.thresh_state]]}".encode('utf-8'))
                time.sleep(0.1)
        except BaseException as e:
            raise e
    
    def update_cycle(self):
        try:
            while not self.stopflag.wait(0.01):
                # print('updating...')
                self.update()
                time.sleep(0.01)
        except BaseException as e:
            raise e
        
    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points)
        num_samples = data.shape[1]
        if num_samples < self.num_points:
            print(f"Not enough data yet: {num_samples}/{self.num_points} samples")
            return
        
        # 3) Filter each EEG channel
        filtered_mu = []
        filtered_beta = []
        for ch in self.eeg_channels:
            mu_band = data[ch].copy() # Mu band is 8-12 Hz
            DataFilter.detrend(mu_band, DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(mu_band, self.sampling_rate, 8.0, 12.0, 4, 
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(mu_band, self.sampling_rate, 48.0, 52.0, 4,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(mu_band, self.sampling_rate, 58.0, 62.0, 4,
                                        FilterTypes.BUTTERWORTH.value, 0)
            
            beta_band = data[ch].copy() # Beta band is 13-30 Hz
            DataFilter.perform_bandstop(beta_band, self.sampling_rate, 13.0, 30.0, 4,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(beta_band, self.sampling_rate, 48.0, 52.0, 4,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(beta_band, self.sampling_rate, 58.0, 62.0, 4,
                                        FilterTypes.BUTTERWORTH.value, 0)
            
            filtered_mu.append(mu_band)
            filtered_beta.append(beta_band)
        
        filtered_emg = []
        for ch in self.emg_channels:
            chdat = data[ch].copy()
            DataFilter.detrend(chdat, DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(chdat, self.sampling_rate, 30.0, 450.0, 4, 
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(chdat, self.sampling_rate, 48.0, 52.0, 4,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(chdat, self.sampling_rate, 58.0, 62.0, 4,
                                        FilterTypes.BUTTERWORTH.value, 0)
            filtered_emg.append(chdat)

        filtered_signals = np.array(filtered_mu + filtered_beta + filtered_emg)  # (250, 6+6+2)
        with open('Models/scaler_save.pkl', 'rb') as f:
            scaler = pickle.load(f)
        filtered_signals = scaler.transform(filtered_signals.T).T

        tensor_in = torch.tensor(filtered_signals, dtype=torch.float32).unsqueeze(0)
        tensor_in = tensor_in.permute(0, 2, 1)     # (1, 250, 14)
        prediction = self.predictor(tensor_in)
        probs = torch.softmax(prediction, dim=1)
        _, pclasses = torch.max(prediction, 1)
        pc = pclasses.detach().numpy()[0]

        prediction = prediction.detach().numpy()[0]
        # print(f"PREDICTION: {prediction}, {pc}, {LABEL_MAP[pc]}")
        print("PROBABILITIES:", probs.detach().numpy()[0], pc)


        dt = 1/self.sampling_rate
        T = 1 # update 1 second
        thr = 0.33
        self.raw_state += dt/T * (STATE_MAP[LABEL_MAP[pc]] - self.raw_state)
        if np.abs(self.raw_state) < thr:
            self.thresh_state = 0
        elif np.abs(self.raw_state) > 1-thr:
            self.thresh_state = np.sign(self.raw_state)
        
        # print(f"PREDICTION: {prediction}")
        # print(f"PREDICTION: {pc}, {LABEL_MAP[pc]}")


if __name__ == '__main__':
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)

    arduino = serial.Serial('/dev/cu.usbmodem101', 9600)
    
    params = BrainFlowInputParams()
    board_id = BoardIds.CYTON_BOARD
    params.serial_port = "/dev/cu.usbserial-DP04W022"
    board_shim = None
    
    try:
        model = EEG_LSTM(14, 128, 2, 3)
        # model = EEGNet(num_channels=16, num_classes=3, samples=250)
        model_path = 'Models/lstm_best_2emg.pth'
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict'])
        print(model)
    except BaseException as e:
        print("Encountered error loading model")
        traceback.print_exc()
        quit()
    
    try:
        board_shim = BoardShim(board_id, params)
        board_shim.prepare_session()
        board_shim.start_stream(450000, '')
        bci = BCI(board_shim, model, arduino)
    except BaseException as e:
        print(f"Encountered error setting up BCI")
        traceback.print_exc()
        quit()
    
    
        
    try:
        while True:
            time.sleep(0.1)
    except BaseException as e:
        print(f"Encountered error reading BCI: {e}")
    finally:
        bci.close()
