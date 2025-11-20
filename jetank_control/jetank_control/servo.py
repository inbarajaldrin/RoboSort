import sys
import os
import time
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

# Dynamically add JETANK path
sys.path.append('/ros2_docker/JETANK')
from SCSCtrl import TTLServo

class JointStatePublisherNode(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')
        self.publisher_ = self.create_publisher(JointState, 'joint_states', 10)
        timer_period = 0.1  # 10Hz
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        # Direction mapping for each servo (1 = normal, -1 = inverted)
        # This accounts for servos that are physically mounted backwards or
        # have inverted control direction. When reading positions, we invert
        # the angle to match the actual joint orientation.
        # Mapping: servo_id -> direction_multiplier
        self.servo_directions = {
            1: -1,  # base_joint (servo 1) - INVERTED
            2: 1,  # shoulder_joint (servo 2) - INVERTED
            3: -1,   # elbow_joint (servo 3)
            4: 1,   # wrist_joint (servo 4)
            5: -1    # camera_joint (servo 5)
        }
        self.get_logger().info('JointState publisher initialized')

    def convert_raw_to_radians(self, raw_pos, servo_id=None):
        """
        Convert raw servo position to radians
        If servo_id is provided, applies direction inversion if needed
        """
        angle_deg = (raw_pos - 512) * (180.0 / 850.0)
        
        # Apply direction inversion if specified for this servo
        if servo_id is not None and servo_id in self.servo_directions:
            angle_deg *= self.servo_directions[servo_id]
        
        angle_rad = angle_deg * math.pi / 180.0
        return angle_rad
    
    def timer_callback(self):
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = [
            'base_joint',
            'shoulder_joint',
            'elbow_joint',
            'wrist_joint',
            'camera_joint'
        ]

        # Read servo positions
        pos1 = TTLServo.infoSingleGet(1)
        pos2 = TTLServo.infoSingleGet(2)
        pos3 = TTLServo.infoSingleGet(3)
        pos4 = TTLServo.infoSingleGet(4)
        pos5 = TTLServo.infoSingleGet(5)

        joint_msg.position = [
            self.convert_raw_to_radians(pos1, servo_id=1),
            self.convert_raw_to_radians(pos2, servo_id=2),
            self.convert_raw_to_radians(pos3, servo_id=3),
            self.convert_raw_to_radians(pos4, servo_id=4),
            self.convert_raw_to_radians(pos5, servo_id=5),
        ]

        self.publisher_.publish(joint_msg)
        
def main(args=None):
    node = None
    try:
        rclpy.init(args=args)
        node = JointStatePublisherNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\nShutting down JointStatePublisher node...')
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()

