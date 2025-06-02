import logging, traceback, csv, threading, time
from datetime import datetime, timezone
from enum import Enum

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

class MODE(Enum):
    REST = 0
    GRIP = 1
    PINCH = 2


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


class BCI:
    def __init__(self, board_shim, model):
        self.board_shim = board_shim
        self.board_id = board_shim.get_board_id()
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        self.accel_channels = BoardShim.get_accel_channels(self.board_id)
        self.timestamp_channel = BoardShim.get_timestamp_channel(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
#        self.window_size = 4
        self.num_points = 250#self.window_size*self.sampling_rate
        self.window_size = self.num_points/self.sampling_rate
        print(f"SETTING WINDOW SIZE TO {self.window_size}")
        self.current_prediction = MODE.REST
        
        self.predictor = model
        
        self.training_mode = False
        self.training_class = MODE.REST
        
        self.stopflag = threading.Event()
        self.thread = threading.Thread(target=self.update_cycle)
        self.thread.start()
    
    def close(self):
        self.stopflag.set()
        time.sleep(0.1)
        self.thread.join()
    
    def update_cycle(self):
        try:
            while not self.stopflag.wait(0.01):
                print('updating...')
                self.update()
        except BaseException as e:
            raise e
        
    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points)
        num_samples = data.shape[1]
        if num_samples < self.num_points:
            print(f"Not enough data yet: {num_samples}/{self.num_points} samples")
            return
#        print(data.shape)
#        print()
        # Filtered data array per channel
        filtered_signals = []
        for ch in self.exg_channels:
            signal = data[ch].copy()
            DataFilter.detrend(signal, DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(signal, self.sampling_rate, 3.0, 45.0, 2, FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(signal, self.sampling_rate, 48.0, 52.0, 2, FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(signal, self.sampling_rate, 58.0, 62.0, 2, FilterTypes.BUTTERWORTH.value, 0)
            filtered_signals.append(signal)
        
        prediction = self.predictor(torch.tensor(data[self.exg_channels+self.accel_channels].T, dtype=torch.float32).unsqueeze(0))
        _, pclasses = torch.max(prediction, 1)
        print(f"CLASSES: {prediction}")
        print(f"PREDICTION: {pclasses}")
#        for i in range(num_samples):
#            self.last_ts = datetime.fromtimestamp(timestamps[i], tz=timezone.utc).isoformat()
#            eeg_values = [filtered_signals[ch_idx][i] for ch_idx in range(len(self.exg_channels))]
#            accel_values = [data[ch][i] for ch in self.accel_channels]
#            row = [ts_iso, self.current_mode] + eeg_values + accel_values
#            self.csv_writer.writerow(row)


if __name__ == '__main__':
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)
    
    params = BrainFlowInputParams()
    board_id = BoardIds.CYTON_BOARD
    params.serial_port = "/dev/cu.usbserial-DP04W022"
    board_shim = None
    
    try:
        model = EEG_LSTM(11, 128, 2, 5)
        model.load_state_dict(torch.load('mlmodels/lstm_best.pth')['model_state_dict'])
        print(model)
    except BaseException as e:
        print("Encountered error loading model")
        traceback.print_exc()
        quit()
    
    try:
        board_shim = BoardShim(board_id, params)
        board_shim.prepare_session()
        board_shim.start_stream(450000, '')
        bci = BCI(board_shim, model)
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
