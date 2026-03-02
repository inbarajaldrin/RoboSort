#!/usr/bin/env python3
# Reference: motor.py (real hardware) — this is the simulation equivalent
"""
Convert /cmd_vel (Twist) to per-wheel velocity commands for Ignition Gazebo.

Replaces DiffDrive because the URDF wheel axes point in opposite directions
(WHEEL_R axis ≈ -Y, WHEEL_L axis ≈ +Y), which DiffDrive can't handle.

Subscribes: /cmd_vel (geometry_msgs/Twist)
Publishes:  /cmd/Revolute_DRIVING_WHEEL_R (std_msgs/Float64)
            /cmd/Revolute_DRIVING_WHEEL_L (std_msgs/Float64)
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64

# Physical layout (URDF names are swapped vs ROS convention):
#   WHEEL_R at y=+0.033 → physical LEFT side,  axis ≈ (0, -1, 0)
#   WHEEL_L at y=-0.055 → physical RIGHT side,  axis ≈ (0, +1, 0)
#
# Axis-to-ground-motion mapping:
#   WHEEL_R (axis -Y): positive velocity → forward
#   WHEEL_L (axis +Y): positive velocity → backward (need to negate)

WHEEL_SEPARATION = 0.1478  # meters (center-to-center)
WHEEL_RADIUS = 0.0273      # meters


class CmdVelToWheels(Node):
    def __init__(self):
        super().__init__('cmd_vel_to_wheels')

        self.pub_wheel_r = self.create_publisher(
            Float64, '/cmd/Revolute_DRIVING_WHEEL_R', 10)
        self.pub_wheel_l = self.create_publisher(
            Float64, '/cmd/Revolute_DRIVING_WHEEL_L', 10)

        self.sub = self.create_subscription(
            Twist, '/cmd_vel', self.on_cmd_vel, 10)

        self.get_logger().info(
            'cmd_vel_to_wheels ready '
            f'(sep={WHEEL_SEPARATION}, radius={WHEEL_RADIUS})')

    def on_cmd_vel(self, msg: Twist):
        # Negate linear_x: wheels_inverted in GUI negates linear for real hardware,
        # but the axis sign correction below already handles forward direction.
        # Un-inverting linear here makes both corrections work together.
        linear_x = -msg.linear.x
        angular_z = msg.angular.z

        # Standard diff drive kinematics (physical left/right)
        phys_left = (linear_x - angular_z * WHEEL_SEPARATION / 2.0) / WHEEL_RADIUS
        phys_right = (linear_x + angular_z * WHEEL_SEPARATION / 2.0) / WHEEL_RADIUS

        # Apply axis sign corrections:
        #   WHEEL_R = physical left,  axis -Y → standard sign
        #   WHEEL_L = physical right, axis +Y → negate
        wheel_r_vel = Float64(data=phys_left)
        wheel_l_vel = Float64(data=-phys_right)

        self.pub_wheel_r.publish(wheel_r_vel)
        self.pub_wheel_l.publish(wheel_l_vel)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelToWheels()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
