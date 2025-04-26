from gpiozero import Motor, DigitalInputDevice
import time

# Motor and encoder setup
M_LEGA = 17
M_LEGB = 27
ENCODER_A = 14  # GPIO pin for encoder A
ENCODER_B = 15  # GPIO pin for encoder B

TICKS_PER_REV = 8  # Number of ticks per revolution for the encoder
GEAR_RATIO = 298  # Gear ratio of the motor

motor = Motor(forward=M_LEGA, backward=M_LEGB, pwm=True)

# Encoder setup
encoder_a = DigitalInputDevice(ENCODER_A, pull_up=True)
encoder_b = DigitalInputDevice(ENCODER_B, pull_up=True)

# Encoder variables
pulse_count = 0
direction = 1  # 1 for forward, -1 for backward
t0 = time.time()

# Encoder pulse callback
def enc_rise():
    global pulse_r
    return 1

def encoder_callback():
    global pulse_count, direction
    # Check the state of Encoder B to determine direction
    if True:# encoder_b.value:
        direction = 1  # Forward
    else:
        direction = -1  # Backward
    pulse_count += direction
    # print(pulse_count, direction)

# Attach the callback to Encoder A's rising edge
encoder_a.when_activated = encoder_callback

# Function to calculate speed
def calculate_speed():
    global pulse_count, t0
    t = time.time()
    dt = t - t0
    t0 = t

    # Calculate revolutions per second (RPS)
    revolutions = abs(pulse_count) / TICKS_PER_REV
    rps = revolutions / dt
    pulse_count = 0  # Reset pulse count after calculation

    # Convert to RPM (Revolutions Per Minute)
    rpm = rps * 60 / GEAR_RATIO
    return rpm, direction

# Run the motor and measure speed
try:
    setspeed = 0.05
    dsp = 0.05
    dirsp = 1
    while True:
        motor.forward(setspeed)
        time.sleep(0.5)  # Measure speed every second
        speed, dir = calculate_speed()
        direction_str = "Forward" if dir == 1 else "Backward"
        print(f"Motor Speed: {speed:.2f} RPM, Direction: {direction_str}")
        setspeed += dirsp * dsp
        if abs(setspeed) < dsp:
            setspeed = dsp
            dirsp = 1
        if abs(setspeed) > 1:
            setspeed = 1
            dirsp = -1
finally:
    motor.stop()

