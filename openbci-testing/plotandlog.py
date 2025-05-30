import argparse
import logging
import csv
from datetime import datetime, timezone

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations


class Graph(QtWidgets.QMainWindow):
    def __init__(self, board_shim):
        super().__init__()
        self.board_shim = board_shim
        self.board_id = board_shim.get_board_id()
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        self.accel_channels = BoardShim.get_accel_channels(self.board_id)
        self.timestamp_channel = BoardShim.get_timestamp_channel(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate
        self.current_mode = "none"

        # Setup UI
        self.app = QtWidgets.QApplication([])
        self.graph_widget = pg.GraphicsLayoutWidget()
        self.setCentralWidget(self.graph_widget)
        self.setWindowTitle("BrainFlow EEG Plot + Logger")
        self.resize(1000, 800)
        self._init_timeseries()

        # Setup Logger
        self.log_file = open("bci_log.csv", "w", newline='')
        self.csv_writer = csv.writer(self.log_file)
        headers = ["timestamp", "mode"] + [f"ch_{ch}" for ch in self.exg_channels] + \
                  [f"accel_{i}" for i in range(3)]
        self.csv_writer.writerow(headers)

        # Timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(self.update_speed_ms)
        self.last_ts = None

        self.show()
        self.app.exec_()

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

    def keyPressEvent(self, event):
        key = event.text().lower()
        if key == 'n':
            self.current_mode = "rest/none"
        elif key == 'l':
            self.current_mode = "left"
        elif key == 'r':
            self.current_mode = "right"
        elif key == 'b':
            self.current_mode = "leg"
        elif key == 'g':
            self.current_mode = "grab"
        print(f"Mode set to: {self.current_mode} at {self.last_ts}")

    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points)
        num_samples = data.shape[1]

        # Filtered data array per channel
        filtered_signals = []
        for ch in self.exg_channels:
            signal = data[ch].copy()
            DataFilter.detrend(signal, DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(signal, self.sampling_rate, 3.0, 45.0, 2, FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(signal, self.sampling_rate, 48.0, 52.0, 2, FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(signal, self.sampling_rate, 58.0, 62.0, 2, FilterTypes.BUTTERWORTH.value, 0)
            filtered_signals.append(signal)

        # Update plot with most recent data
        for i, signal in enumerate(filtered_signals):
            self.curves[i].setData(signal.tolist())

        timestamps = data[self.timestamp_channel]

        for i in range(num_samples):
            ts_iso = datetime.fromtimestamp(timestamps[i], tz=timezone.utc).isoformat()
            self.last_ts = ts_iso
#datetime.utcfromtimestamp(timestamps[i]).isoformat()
            eeg_values = [filtered_signals[ch_idx][i] for ch_idx in range(len(self.exg_channels))]
            accel_values = [data[ch][i] for ch in self.accel_channels]
            row = [ts_iso, self.current_mode] + eeg_values + accel_values
            self.csv_writer.writerow(row)

        self.app.processEvents()

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

