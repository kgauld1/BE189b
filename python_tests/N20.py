import time
from gpiozero import Motor, DigitalInputDevice

# Motor and encoder setup
M_LEGA = 17
M_LEGB = 27
ENCODER_A = 14  # GPIO pin for encoder A
ENCODER_B = 15  # GPIO pin for encoder B

class N20:
    def __init__(self, legA, legB, encA, encB, ticks_per_rev=8, gear_ratio=298):
        self.motor = Motor(forward=legA, backward=legB)
        self.encA = DigitalInputDevice(encA, pull_up=True)
        self.encB = DigitalInputDevice(encB, pull_up=True)
        
        self.count = 0

        self.ticks_per_rev = ticks_per_rev
        self.gear_ratio = gear_ratio

        self.direction = 0
        self.speed = 0
        self.position = 0

        self.t_rise = None

        self.encA.when_activated = self.enc_A_rise
        self.encB.when_deactivated = self.enc_A_fall
    
    def forward(self, setspeed=1):
        self.motor.forward(setspeed)
    def backward(self, setspeed=1):
        self.motor.backward(setspeed)
    def stop(self):
        self.motor.stop()
    
    def enc_A_rise(self):
        if self.encB.value:
            self.direction = -1
        else:
            self.direction = 1
        self.count += self.direction

        self.t_rise = time.time()

    def enc_A_fall(self):
        if self.t_rise is None: return
        T = (time.time()-self.t_rise)/60
        self.speed = 1/(self.gear_ratio*self.ticks_per_rev*T)

        self.position = self.count/(self.ticks_per_rev*self.gear_ratio)
    
if __name__ == "__main__":
    motor = N20(M_LEGA, M_LEGB, ENCODER_A, ENCODER_B)
    try:
        setspeed = 0.05
        dsp = 0.05
        dirsp = 1
        while True:
            motor.forward(setspeed)
            time.sleep(0.5)
            print(f"Speed: {motor.speed:.2f} RPM, Position: {motor.position} revs")
            setspeed += dirsp*dsp
            if abs(setspeed) < dsp:
                setspeed = dsp
                dirsp = 1
            if abs(setspeed) > 1:
                setspeed = 1
                dirsp = -1
    finally:
        motor.stop()



