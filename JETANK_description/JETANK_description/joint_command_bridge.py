#!/usr/bin/env python3
# Reference: bcr_bot JointPositionController pattern (/tmp/bcr_bot)
"""
Bridge node: splits /joint_commands (JointState) into individual Float64 topics
that Ignition Gazebo JointPositionController plugins listen on.

Each joint gets a topic: /cmd/<joint_name> (std_msgs/Float64)
These are bridged to Ignition via ros_gz_bridge (ROS2 → Gz direction).
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

CONTROLLED_JOINTS = [
    'revolute_BEARING',
    'Revolute_SERVO_LOWER',
    'Revolute_SERVO_UPPER',
    'revolute_CAMERA_HOLDER_ARM_LOWER',
    'revolute_GRIPPER_L1',
    'revolute_GRIPPER_L2',
    'Revolute_GRIPPER_R1',
    'Revolute_GRIPPER_R2',
    # Free wheels: roll freely, no control needed
    # Drive wheels: handled by DiffDrive plugin via /cmd_vel
]


class JointCommandBridge(Node):
    def __init__(self):
        super().__init__('joint_command_bridge')
        self.pubs = {}
        for name in CONTROLLED_JOINTS:
            self.pubs[name] = self.create_publisher(Float64, f'/cmd/{name}', 10)
        self.sub = self.create_subscription(
            JointState, '/joint_commands', self.on_cmd, 10)
        self.get_logger().info(
            f'Joint command bridge ready for {len(CONTROLLED_JOINTS)} joints')

    def on_cmd(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            if name in self.pubs:
                out = Float64()
                out.data = pos
                self.pubs[name].publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = JointCommandBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
