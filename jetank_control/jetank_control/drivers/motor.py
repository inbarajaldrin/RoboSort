#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from Adafruit_MotorHAT import Adafruit_MotorHAT


class MotorDriver(Node):
    def __init__(self):
        super().__init__('MotorDriver')
        self.subscription = self.create_subscription(Twist, 'cmd_vel', self.twist_callback, 10)
        
        # Motor IDs
        self.MOTOR_LEFT = 1
        self.MOTOR_RIGHT = 2
        
        # Initialize hardware
        self.driver = Adafruit_MotorHAT(i2c_bus=1)
        self.motors = {
            self.MOTOR_LEFT: self.driver.getMotor(self.MOTOR_LEFT),
            self.MOTOR_RIGHT: self.driver.getMotor(self.MOTOR_RIGHT)
        }
        self.pwm_channels = {
            self.MOTOR_LEFT: (1, 0),
            self.MOTOR_RIGHT: (2, 3)
        }
        
        # Track last command
        self.last_left = -999
        self.last_right = -999
        
        self.get_logger().info('Motor driver initialized')
    
    def twist_callback(self, msg):
        """Convert Twist message to motor speeds"""
        linear_x = msg.linear.x
        angular_z = msg.angular.z
        
        # Differential drive
        left = linear_x - angular_z
        right = linear_x + angular_z
        
        # Clamp to [-1, 1]
        left = max(min(left, 1.0), -1.0)
        right = max(min(right, 1.0), -1.0)
        
        # Skip if no change
        if left == self.last_left and right == self.last_right:
            return
        
        self.last_left = left
        self.last_right = right
        
        # Send to motors
        self.set_motor_speed(self.MOTOR_LEFT, left)
        self.set_motor_speed(self.MOTOR_RIGHT, right)
    
    def set_motor_speed(self, motor_id, value):
        """Set motor speed between [-1.0, 1.0]"""
        # Convert to PWM (0-255)
        pwm = int(abs(value) * 255)
        
        self.motors[motor_id].setSpeed(pwm)
        
        # Get PWM channels
        ina, inb = self.pwm_channels[motor_id]
        
        # Set direction
        if value > 0:
            self.motors[motor_id].run(Adafruit_MotorHAT.FORWARD)
            self.driver._pwm.setPWM(ina, 0, pwm * 16)
            self.driver._pwm.setPWM(inb, 0, 0)
        elif value < 0:
            self.motors[motor_id].run(Adafruit_MotorHAT.BACKWARD)
            self.driver._pwm.setPWM(ina, 0, 0)
            self.driver._pwm.setPWM(inb, 0, pwm * 16)
        else:
            self.motors[motor_id].run(Adafruit_MotorHAT.RELEASE)
            self.driver._pwm.setPWM(ina, 0, 0)
            self.driver._pwm.setPWM(inb, 0, 0)
    
    def stop(self):
        """Stop all motors"""
        self.set_motor_speed(self.MOTOR_LEFT, 0.0)
        self.set_motor_speed(self.MOTOR_RIGHT, 0.0)
    
    def destroy_node(self):
        """Clean shutdown"""
        self.get_logger().info('Stopping motors')
        self.stop()
        super().destroy_node()


def main(args=None):
    driver = None
    try:
        rclpy.init(args=args)
        driver = MotorDriver()
        rclpy.spin(driver)
    except KeyboardInterrupt:
        print('\nShutting down motor driver...')
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
    finally:
        if driver is not None:
            driver.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
