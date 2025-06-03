import logging
import csv
import random
import time
from datetime import datetime, timezone

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets,QtCore

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations


class Graph(QtWidgets.QMainWindow):
    def __init__(self, board_shim):
        super().__init__()
        self.board_shim = board_shim
        self.board_id = board_shim.get_board_id()
        self.exg_channels = board_shim.get_exg_channels(self.board_id)
        self.eeg_channels = self.exg_channels[:-2]
        self.emg_channels = self.exg_channels[-2:]
        self.accel_channels = board_shim.get_accel_channels(self.board_id)
        self.timestamp_channel = board_shim.get_timestamp_channel(self.board_id)
        self.sampling_rate = board_shim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate
        
        # Only “close” and “pinch” are chosen during active; “rest” is used for lead-in/lead-out.
        self.available_modes = ['close-ARM', 'pinch-LEG']

        self.mode_counts = {
            'close-ARM': 0,
            'pinch-LEG': 0
        }

        self.state = 'lead_in'
        self.state_start_time = time.time()
        self.trial_duration = self._pick_random_duration()
        self.trial_target_mode = random.choice(self.available_modes)
        self.current_mode = "rest"

        # Setup UI
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QtWidgets.QVBoxLayout()
        self.central_widget.setLayout(layout)

        self.instruction_label = QtWidgets.QLabel(f"Rest (Coming up: {self.trial_target_mode})")
        self.instruction_label.setAlignment(QtCore.Qt.AlignCenter)
        font = self.instruction_label.font()
        font.setPointSize(24)
        self.instruction_label.setFont(font)
        layout.addWidget(self.instruction_label)
        
        self.graph_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graph_widget)
        self.setWindowTitle("BrainFlow EEG Plot + Automated Trials (Close/Pinch)")
        self.resize(1000, 800)
        self._init_timeseries()

        # CSV Logger
        self.log_file = open("bci_log.csv", "w", newline='')
        self.csv_writer = csv.writer(self.log_file)
        headers = (
            ["timestamp", "mode"]
            + [f"ch_{ch}_mu" for ch in self.eeg_channels]
            + [f"ch_{ch}_beta" for ch in self.eeg_channels]
            + [f"ch_{ch}_emg" for ch in self.emg_channels]
            + [f"accel_{i}" for i in range(3)]
        )
        self.csv_writer.writerow(headers)

        # Timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(self.update_speed_ms)
        self.newest_ts = None

    def _pick_random_duration(self):
        return random.uniform(7.8, 8.2)
    
    def _init_timeseries(self):
        self.plots = []
        self.curves = []
        for i in range(len(self.exg_channels)):
            p = self.graph_widget.addPlot(row=i, col=0)
            p.showAxis('left', False)
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            if i == 0:
                p.setTitle('Filtered EEG')
            curve = p.plot()
            self.plots.append(p)
            self.curves.append(curve)

    def _advance_state_if_needed(self):
        """
        Manage lead_in (4s), active (Ns), lead_out (4s), then loop.
        """
        now = time.time()
        elapsed = now - self.state_start_time

        if self.state == "lead_in":
            if elapsed >= 2.0:
                self.state = "active"
                self.state_start_time = now
                self.current_mode = self.trial_target_mode
                self.instruction_label.setText(f"{self.trial_target_mode}")

        elif self.state == "active":
            if elapsed >= self.trial_duration:
                self.state = "lead_out"
                self.state_start_time = now
                self.current_mode = "rest"
                self.instruction_label.setText("Rest")

        elif self.state == "lead_out":
            if elapsed >= 2.0:
                self.state = "lead_in"
                self.state_start_time = now
                self.trial_duration = self._pick_random_duration()

                if self.mode_counts['close-ARM']-self.mode_counts['pinch-LEG'] > 3:
                    self.trial_target_mode = 'pinch-LEG'
                elif self.mode_counts['pinch-LEG']-self.mode_counts['close-ARM'] > 3:
                    self.trial_target_mode = 'close-ARM'
                else:
                    self.trial_target_mode = random.choice(self.available_modes)
                
                self.mode_counts[self.trial_target_mode] += 1
                self.current_mode = "rest"
                self.instruction_label.setText(f"Rest (Coming up: {self.trial_target_mode})")

    def update(self):
        # 1) Possibly advance lead_in → active → lead_out
        self._advance_state_if_needed()

        # 2) Grab real data from board
        data = self.board_shim.get_current_board_data(self.num_points)
        num_samples = data.shape[1]

        # 3) Filter each EEG channel
        filtered_mu = []
        filtered_beta = []
        for ch in self.eeg_channels:
            mu_band = data[ch].copy() # Mu band is 8-12 Hz
            DataFilter.detrend(mu_band, DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(mu_band, self.sampling_rate,  8.0, 12.0, 4, 
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
        
        emg_data = []
        for ch in self.emg_channels:
            chdata = data[ch].copy()
            DataFilter.detrend(chdata, DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(chdata, self.sampling_rate, 30.0, 450.0, 4, 
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(chdata, self.sampling_rate, 48.0, 52.0, 4,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(chdata, self.sampling_rate, 58.0, 62.0, 4,
                                        FilterTypes.BUTTERWORTH.value, 0)
            emg_data.append(chdata)

        # 4) Update the curves on screen
        for i, sig in enumerate(filtered_beta):
            self.curves[i].setData(sig)
        for i, sig in enumerate(emg_data):
            self.curves[i+len(filtered_beta)].setData(sig)
        
        # 5) Log every sample: timestamp, mode, eeg channel, accel channel
        timestamps = data[self.timestamp_channel]
        for i in range(num_samples):
            ts_iso = datetime.fromtimestamp(timestamps[i], tz=timezone.utc).isoformat()
            if self.newest_ts is not None and ts_iso <= self.newest_ts:
                continue
            self.newest_ts = ts_iso
            eeg_mu_vals = [
                filtered_mu[ch_idx][i]
                for ch_idx in range(len(self.eeg_channels))
            ]
            eeg_beta_vals = [
                filtered_beta[ch_idx][i]
                for ch_idx in range(len(self.eeg_channels))
            ]
            emg_vals = [emg_data[ch][i] for ch in range(len(self.emg_channels))]
            accel_vals = [data[ch][i] for ch in self.accel_channels]
            row = [ts_iso, self.current_mode] + eeg_mu_vals + eeg_beta_vals + emg_vals + accel_vals
            self.csv_writer.writerow(row)

        # 6) Process Qt events
        QtWidgets.QApplication.instance().processEvents()

    def closeEvent(self, event):
        self.log_file.close()
        super().closeEvent(event)

def main():
    import sys
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)

    app = QtWidgets.QApplication(sys.argv)  # Create QApplication FIRST

    params = BrainFlowInputParams()
    board_id = BoardIds.CYTON_BOARD
    params.serial_port = "/dev/cu.usbserial-DP04W022"

    board_shim = None
    window = None
    try:
        board_shim = BoardShim(board_id, params)
        board_shim.prepare_session()
        board_shim.start_stream(450000, '')
        
        window = Graph(board_shim)  # No QApplication creation inside Graph
        window.show()
        app.exec_()                 # Run the Qt event loop here
        
    except BaseException:
        logging.warning('Exception', exc_info=True)
    finally:
        logging.info('End')
        if board_shim is not None and board_shim.is_prepared():
            logging.info('Releasing session')
            board_shim.release_session()
            
if __name__ == '__main__':
    main()

