import time
import numpy as np
import matplotlib.pyplot as plt
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# Parameters - change these for your setup
serial_port = "/dev/tty.usbserial-DP04W022"#"/dev/ttyUSB0"  # or "COM3" on Windows
board_id = BoardIds.CYTON_BOARD.value  # Cyton board ID
sampling_rate = 250  # Cyton default

# Initialize BrainFlow input parameters
params = BrainFlowInputParams()
params.serial_port = serial_port

# Create board object
board = BoardShim(board_id, params)

# Select EEG channel to plot (e.g., channel 1)
eeg_channel = 1  # OpenBCI Cyton EEG channels 1-8 are indexed 1 to 8 in BrainFlow

# Setup plot
plt.ion()
fig, ax = plt.subplots()
x_len = 500  # number of points to display
y_data = np.zeros(x_len)
x_data = np.arange(x_len)
line, = ax.plot(x_data, y_data)
ax.set_ylim(-100, 100)  # Adjust based on expected EEG amplitude
ax.set_xlabel("Samples")
ax.set_ylabel("EEG amplitude (uV)")
ax.set_title("Live EEG from OpenBCI Cyton")

try:
    board.prepare_session()
    board.start_stream(45000)  # optional to save to file
    print("Streaming started...")

    while True:
        # Get board data - rows: channels, columns: samples
        data = board.get_board_data()
        if data.shape[1] == 0:
            continue  # no data yet

        # Get the EEG channel data - channel indices start at 0
        # BrainFlow channel mapping for Cyton board:
        # EEG channels are from index 1 to 8 (1-based)
        eeg_data = data[eeg_channel, :]

        # Update plot data (take last x_len samples)
        if len(eeg_data) >= x_len:
            y_data = eeg_data[-x_len:]
        else:
            y_data = np.pad(eeg_data, (x_len - len(eeg_data), 0), mode='constant')

        line.set_ydata(y_data)
        plt.pause(0.01)  # small pause to update plot

except KeyboardInterrupt:
    print("Stopping streaming...")

finally:
    board.stop_stream()
    board.release_session()
    plt.ioff()
    plt.show()
