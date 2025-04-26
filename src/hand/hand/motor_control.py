import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32


from gpiozero import Motor, DigitalInputDevice
import time

# Motor and encoder setup
M_LEGA, M_LEGB = 17, 27
ENC_A, ENC_B = 14, 15


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

        self.t_last = None

        self.encA.when_activated = self.enc_A_rise
    
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

        if self.t_last is not None:
            T = (time.time()-self.t_last)/60
            self.speed = self.direction/(self.gear_ratio*self.ticks_per_rev*T)
            self.position = self.count/(self.ticks_per_rev*self.gear_ratio)
        self.t_last = time.time()

class MotorController(Node):
    def __init__(self):
        super().__init__('motor_controller')
        self.motor = N20(M_LEGA, M_LEGB, ENC_A, ENC_B)
        self.setpoint = 0

        self.kp = 0.01
        self.ki = 0#0.01
        self.kd = 0#0.005

        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = self.get_clock().now().nanoseconds / 1e9

        self.control_output = 0.0

        self.vel_cb = self.create_subscription(
            Float32,
            'vel_setpoint',
            self.setpoint_cb,
            10)
        
        self.vel_pub = self.create_publisher(Float32, 'actual_vel', 10)

        self.create_timer(1, self.control_loop)
        self.get_logger().info('Motor controller started')

    def setpoint_cb(self, msg):
        self.setpoint = msg.data

    def control_loop(self):
        # Calculate the error
        current_time = self.get_clock().now().nanoseconds / 1e9
        dt = current_time - self.last_time
        if dt == 0: return

        error = self.setpoint - self.motor.speed
        self.integral += error * dt
        derivative = (error - self.last_error) / dt

        # Calculate the control output
        d_out = (
            self.kp * error +
            self.ki * self.integral +
            self.kd * derivative
        )
        self.control_output = max(min(self.control_output + d_out, 1), -1)
        self.last_error = error

        # Set the motor speed
        if self.control_output > 0:
            self.motor.forward(self.control_output)
        else:
            self.motor.backward(-self.control_output)

        msg = Float32()
        msg.data = float(self.motor.speed)
        self.vel_pub.publish(msg)
        # Print the current position and speed
        self.get_logger().info(f'Error: {error}, Setpoint: {self.setpoint}, Integral: {self.integral}, Speed: {self.motor.speed}')

def main(args=None):
    rclpy.init(args=args)
    node = MotorController()
    rclpy.spin(node)
    # Destroy the node explicitly
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()