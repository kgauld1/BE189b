#from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
#
#BoardShim.enable_dev_board_logger()
#
#params = BrainFlowInputParams()
#params.serial_port = "/dev/cu.usbserial-DP04W022"  # Use the correct port for your system
#
#print("READING BOARD")
#board = BoardShim(BoardIds.CYTON_BOARD, params)


import numpy as np
import matplotlib.pyplot as plt
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# Setup
BoardShim.enable_dev_board_logger()

params = BrainFlowInputParams()
params.serial_port = "/dev/cu.usbserial-DP04W022"  # Adjust as needed
board = BoardShim(BoardIds.CYTON_BOARD, params)
board.prepare_session()

board.start_stream()
#time.sleep(1)

#for ch in range(1, 9):
#    board.config_board(f'x{ch}000000X')  # Turn off test signal on each channel
#board.config_board('0')  # Channel 1: default mode, test signal off

# Channel info
eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)
sampling_rate = BoardShim.get_sampling_rate(BoardIds.CYTON_BOARD.value)
window_size = 5  # seconds
num_points = sampling_rate * window_size

# Setup plot
#plt.ion()
fig, axs = plt.subplots(len(eeg_channels), 1, figsize=(10, 8), sharex=True)
lines = []
timestamps = np.linspace(-window_size, 0, num_points)

#for i, ch in enumerate(eeg_channels):
##    axs[i].set_ylim(-100, 100)
#    axs[i].set_xlim(-window_size, 0)
#    axs[i].set_ylabel(f"Ch {ch}")
#    line, = axs[i].plot(timestamps, np.zeros(num_points))
#    lines.append(line)

#axs[-1].set_xlabel("Time (s)")

# Live update loop
try:
    while True:
        data = board.get_current_board_data(num_points)
#        if data.shape[1] < num_points:
#            padded = np.zeros((data.shape[0], num_points))
#            padded[:, -data.shape[1]:] = data
#            data = padded
        
        for i in range(8):
            axs[i].cla()
            axs[i].plot(data[22], data[i+1])
            
        plt.pause(0.01)

except KeyboardInterrupt:
    print("Stopped.")
finally:
    board.stop_stream()
    board.release_session()
