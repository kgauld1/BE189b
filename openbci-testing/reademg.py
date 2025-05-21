import socket
import sys
import time
import argparse
import signal
import struct
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import serial

numSamples = 0

# Print received message to console
def print_message(*args):
    try:
        print(args[0]) #added to see raw data
        obj = json.loads(args[0].decode())
        print(obj.get('data'))
        print("data")
        numSamplesInChannelOne = len(obj.get('data')[0])
        global numSamples
        print("NumSamplesInPacket_ChannelOne == " + str(numSamplesInChannelOne))
        print(f"Num channels = {len(obj.get('data'))}")
        numSamples += numSamplesInChannelOne
        if obj:
            return True
        else:
            return False
    except BaseException as e:
        print(e)
        return False
 #  print("(%s) RECEIVED MESSAGE: " % time.time() +
 # ''.join(str(struct.unpack('>%df' % int(length), args[0]))))

def plot_message(record, data):
    try:
        # print(args[0]) #added to see raw data
        obj = json.loads(data.decode())
        reading = obj.get('data')
        for channel_idx in range(len(record)):
            record[channel_idx].extend(reading[channel_idx])
        numSamplesInChannelOne = len(obj.get('data')[0])
        global numSamples
#        print("NumSamplesInPacket_ChannelOne == " + str(numSamplesInChannelOne))
#        print(f"Num channels = {len(obj.get('data'))}")
        numSamples += numSamplesInChannelOne
        if obj:
            return True
        else:
            return False
    except BaseException as e:
        print(e)
        return False

# Clean exit from print mode
def exit_print(signal, frame):
    print("Closing listener")
    sys.exit(0)

# Record received message in text file
def record_to_file(*args):
    textfile.write(str(time.time()) + ",")
    #textfile.write(args[0])
    obj = json.loads(args[0].decode())
    #print(obj.get('data'))
    textfile.write(json.dumps(obj))
    textfile.write("\n")

# Save recording, clean exit from record mode
def close_file(*args):
    print("\nFILE SAVED")
    textfile.close()
    sys.exit(0)

if __name__ == "__main__":
  # Collect command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument("--ip",
      default="127.0.0.1", help="The ip to listen on")
  parser.add_argument("--port",
      type=int, default=12345, help="The port to listen on")
  parser.add_argument("--address",default="/openbci", help="address to listen to")
  parser.add_argument("--option",default="print",help="Debugger option")
  parser.add_argument("--len",default=9,help="Debugger option")
  args = parser.parse_args()
  
  ser = serial.Serial('/dev/cu.usbmodem1101', 9600)

  # Set up necessary parameters from command line
  length =  int(args.len)
  if args.option=="print":
      signal.signal(signal.SIGINT, exit_print)
  elif args.option=="record":
      i = 0
      while os.path.exists("udp_test%s.txt" % i):
        i += 1
      filename = "udp_test%i.txt" % i
      textfile = open(filename, "w")
      textfile.write("time,address,messages\n")
      textfile.write("-------------------------\n")
      print("Recording to %s" % filename)
      signal.signal(signal.SIGINT, close_file)

  # Connect to socket
  sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  server_address = (args.ip, args.port)
  sock.bind(server_address)

  # Display socket attributes
  print('--------------------')
  print("-- UDP LISTENER -- ")
  print('--------------------')
  print("IP:", args.ip)
  print("PORT:", args.port)
  print('--------------------')
  print("%s option selected" % args.option)

  # Receive messages
  print("Listening...")
  start = time.time()
  
  duration = float('inf')
  timestamps = []
  records = [[],[],[],[],[],[],[],[]]
  
  
  try:
    while time.time() <= start + duration:
        data, addr = sock.recvfrom(20000) # buffer size is 20000 bytes
        plot_message(records, data)
        R_per_CH = len(records[0])
        ch_7_hist = np.std(records[6][-200:])
        ch_8_hist = np.std(records[7][-200:])
        
        if ch_7_hist < 35 and ch_8_hist < 50:
            mode = "relaxed"
            ser.write('1'.encode('utf-8'))
        elif ch_8_hist < 80:
            mode = "pinch"
            ser.write('2'.encode('utf-8'))
        else:
            mode = "flex"
            ser.write('3'.encode('utf-8'))
        numSamples += 1
        if args.option=="plot":
            plt.clf()
            try:
                plt.gca().get_legend().remove()
            except:
                pass
            for channel_idx in range(6,8):
                plt.plot(np.linspace(start, time.time(), num=R_per_CH),np.array(records[channel_idx])-channel_idx*400, label=f'ch {channel_idx+1} {np.std(records[channel_idx][-200:]):0.2f}')
            plt.plot([],[],label=f'MODE: {mode}')
            plt.legend(loc='upper right')
            plt.pause(0.001)
        elif args.option=="record":
            record_to_file(data)
            numSamples += 1
        else:
            print(f"CH7: {ch_7_hist:0.1f}\t CH8: {ch_8_hist:0.1f}")
            print(f"MODE: {mode}")
  except BaseException as e:
    print('Exiting...')

print( "Samples == {}".format(numSamples) )
print( "Duration == {}".format(duration) )
print( "Avg Sampling Rate == {}".format(numSamples / duration) )

