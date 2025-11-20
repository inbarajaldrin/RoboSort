#!/usr/bin/env python3
"""
Servo Driver
Subscribes to joint_commands and controls servos.
Publishes joint_states feedback from servo positions.
"""

import sys
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

# Dynamically add JETANK path
sys.path.append('/ros2_docker/JETANK')
from SCSCtrl import TTLServo


class ServoDriver(Node):
    """
    ROS2 node for controlling servos and publishing joint states.
    Subscribes to joint_commands and publishes current joint_states.
    """
    
    def __init__(self):
        super().__init__('ServoDriver')
        
        # Subscribe to joint commands from arm_controller
        self.subscription = self.create_subscription(
            JointState,
            'joint_commands',
            self.joint_command_callback,
            10
        )
        
        # Publisher for joint states feedback
        self.publisher = self.create_publisher(JointState, 'joint_states', 10)
        
        # Timer to read servos at 10Hz
        self.timer = self.create_timer(0.1, self.publish_joint_states)
        
        # Joint names mapping
        self.joint_names = [
            'base_joint',
            'shoulder_joint',
            'elbow_joint',
            'wrist_joint',
            'camera_joint'
        ]
        
        # Servo ID mapping (1-5 correspond to the joints)
        self.servo_ids = [1, 2, 3, 4, 5]
        
        # Direction mapping for each servo (1 = normal, -1 = inverted)
        # MUST match the direction mapping in arm_controller.py
        self.servo_directions = {
            1: -1,  # base_joint (servo 1) - INVERTED
            2: 1,   # shoulder_joint (servo 2) - normal
            3: -1,  # elbow_joint (servo 3) - INVERTED
            4: 1,   # wrist_joint (servo 4) - normal
            5: -1   # camera_joint (servo 5) - INVERTED
        }
        
        # Default speed for servo movements (1-150, higher = faster)
        self.default_speed = 150
        
        self.get_logger().info('Servo driver initialized')
        self.get_logger().info('Listening on topic: joint_commands')
        self.get_logger().info('Publishing: joint_states')
    
    def convert_radians_to_degrees(self, angle_rad):
        """Convert angle in radians to degrees"""
        return angle_rad * 180.0 / math.pi
    
    def convert_raw_to_radians(self, raw_pos, servo_id):
        """Convert raw servo position (0-1024) to radians"""
        # Convert raw position to degrees
        angle_deg = (raw_pos - 512) * (180.0 / 850.0)
        
        # Apply direction inversion if needed
        angle_deg *= self.servo_directions[servo_id]
        
        # Convert to radians
        angle_rad = angle_deg * math.pi / 180.0
        return angle_rad
    
    def get_speed_from_velocity(self, velocity_rad_s):
        """Convert velocity in rad/s to servo speed (1-150)"""
        if velocity_rad_s is None or velocity_rad_s == 0.0:
            return self.default_speed
        
        # Convert rad/s to deg/s
        velocity_deg_s = abs(velocity_rad_s) * 180.0 / math.pi
        
        # Map velocity to speed (1-150)
        # Assuming max velocity of ~300 deg/s maps to speed 150
        speed = int(max(1, min(150, velocity_deg_s * 150.0 / 300.0)))
        return speed
    
    def joint_command_callback(self, msg):
        """Handle joint commands and control servos"""
        if not msg.name:
            self.get_logger().warn('Received empty joint command')
            return
        
        # Create mappings of joint names to values
        joint_positions = {}
        joint_velocities = {}
        
        if msg.position:
            joint_positions = dict(zip(msg.name, msg.position))
        if msg.velocity:
            joint_velocities = dict(zip(msg.name, msg.velocity))
        
        # Apply commands to servos
        for i, joint_name in enumerate(self.joint_names):
            if joint_name in joint_positions:
                angle_rad = joint_positions[joint_name]
                angle_deg = self.convert_radians_to_degrees(angle_rad)
                
                servo_id = self.servo_ids[i]
                
                # Get velocity/speed if provided
                velocity = joint_velocities.get(joint_name, None)
                speed = self.get_speed_from_velocity(velocity)
                
                # Get direction for this servo
                direction = self.servo_directions.get(servo_id, 1)
                
                try:
                    # Control servo with angle, direction, and speed
                    TTLServo.servoAngleCtrl(servo_id, int(angle_deg), direction, speed)
                    self.get_logger().debug(
                        f'Set {joint_name} (servo {servo_id}) to {angle_deg:.1f}° at speed {speed}'
                    )
                except Exception as e:
                    self.get_logger().error(f'Failed to set servo {servo_id}: {str(e)}')
    
    def publish_joint_states(self):
        """Read servo positions and publish joint states"""
        try:
            # Create joint state message
            joint_msg = JointState()
            joint_msg.header.stamp = self.get_clock().now().to_msg()
            joint_msg.name = self.joint_names
            
            # Read positions from all servos
            positions = []
            for servo_id in self.servo_ids:
                raw_pos = TTLServo.infoSingleGet(servo_id)
                angle_rad = self.convert_raw_to_radians(raw_pos, servo_id)
                positions.append(angle_rad)
            
            joint_msg.position = positions
            
            # Publish
            self.publisher.publish(joint_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error reading servo positions: {e}')
    
    def destroy_node(self):
        """Clean shutdown"""
        self.get_logger().info('Stopping servo driver')
        super().destroy_node()


def main(args=None):
    driver = None
    try:
        rclpy.init(args=args)
        driver = ServoDriver()
        rclpy.spin(driver)
    except KeyboardInterrupt:
        print('\nShutting down servo driver...')
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

