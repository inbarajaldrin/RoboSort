#!/usr/bin/env python3
# Reference: motor.py (real hardware) — this is the simulation equivalent
"""
Convert /cmd_vel (Twist) to per-wheel velocity commands for Ignition Gazebo.

SUPERSEDED (as of the 02e642f drift fix): gazebo.launch.py no longer launches
this node — the Ignition DiffDrive plugin in JETANK.gazebo now consumes /cmd_vel
directly. That became possible once the drive-wheel joint axes were normalized to
a common (0 1 0); previously WHEEL_R was the opposing (0 -1 0) axis, which the
stock DiffDrive couldn't handle, which is why this custom bridge existed. Kept as
a reference / for non-DiffDrive setups, and updated to the new same-axis
convention so it stays correct.

Subscribes: /cmd_vel (geometry_msgs/Twist)
Publishes:  /cmd/Revolute_DRIVING_WHEEL_R (std_msgs/Float64)
            /cmd/Revolute_DRIVING_WHEEL_L (std_msgs/Float64)
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64

# Physical layout (URDF names are swapped vs ROS convention):
#   WHEEL_R at y=+0.043 → physical LEFT side,  axis (0, 1, 0)
#   WHEEL_L at y=-0.043 → physical RIGHT side, axis (0, 1, 0)
#
# Axis-to-ground-motion mapping (post-02e642f, both axes normalized to (0 1 0)):
#   positive joint velocity → wheel rolls FORWARD, for BOTH wheels.
#   So there is no per-wheel sign correction — both publish their standard
#   diff-drive angular velocity directly. (Old code negated WHEEL_L to
#   compensate for its then-opposing axis; that negation is now removed.)

WHEEL_SEPARATION = 0.1240  # meters (center-to-center, symmetric layout)
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
        # +linear.x = forward. Both wheel axes are (0 1 0) → positive joint
        # velocity rolls each wheel forward, so no input/output sign flips.
        linear_x = msg.linear.x
        angular_z = msg.angular.z

        # Standard diff drive kinematics (physical left/right wheel angular vel)
        phys_left = (linear_x - angular_z * WHEEL_SEPARATION / 2.0) / WHEEL_RADIUS
        phys_right = (linear_x + angular_z * WHEEL_SEPARATION / 2.0) / WHEEL_RADIUS

        # WHEEL_R = physical left, WHEEL_L = physical right; both axes (0 1 0),
        # so each publishes its standard sign directly (no negation).
        wheel_r_vel = Float64(data=phys_left)
        wheel_l_vel = Float64(data=phys_right)

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
