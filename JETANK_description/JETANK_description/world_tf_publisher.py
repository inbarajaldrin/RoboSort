#!/usr/bin/env python3
"""Publish world -> base_footprint TF from Gazebo model pose.

Queries the actual jetank model pose from Gazebo's dynamic_pose topic
and publishes it as a TF transform. This closes the gap between the
Gazebo world frame and the ROS2 TF tree (which starts at base_footprint).

Without this, any node that needs world-frame coordinates (e.g.
verify_detections, randomize_legos) must assume the robot hasn't drifted
from its spawn position — which breaks after sim drift.
"""
import re
import subprocess

import rclpy
from rclpy.node import Node
import tf2_ros
from geometry_msgs.msg import TransformStamped


def _parse_jetank_pose(text):
    """Extract jetank model pose from Gazebo dynamic_pose protobuf text."""
    blocks = re.split(r'(?=^pose\s*\{)', text, flags=re.MULTILINE)
    for block in blocks:
        if '"jetank"' not in block:
            continue

        def _val(txt, key):
            m = re.search(rf'{key}:\s*([-\d.e+]+)', txt)
            return float(m.group(1)) if m else 0.0

        pos_m = re.search(r'position\s*\{([^}]*)\}', block)
        ori_m = re.search(r'orientation\s*\{([^}]*)\}', block)
        pb = pos_m.group(1) if pos_m else ""
        ob = ori_m.group(1) if ori_m else ""
        pos = (_val(pb, 'x'), _val(pb, 'y'), _val(pb, 'z'))
        quat = (_val(ob, 'x'), _val(ob, 'y'), _val(ob, 'z'), _val(ob, 'w'))
        return pos, quat

    return None, None


class WorldTFPublisher(Node):
    def __init__(self):
        super().__init__('world_tf_publisher')
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        # Query at 2Hz — drift is slow, no need for high rate
        self.timer = self.create_timer(0.5, self._publish_tf)
        self.get_logger().info('Publishing world -> base_footprint TF from Gazebo')

    def _publish_tf(self):
        try:
            result = subprocess.run(
                ["ign", "topic", "-e", "-t",
                 "/world/lego_world/dynamic_pose/info", "-n", "1"],
                capture_output=True, text=True, timeout=5
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return

        pos, quat = _parse_jetank_pose(result.stdout)
        if pos is None:
            return

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'base_footprint'
        t.transform.translation.x = pos[0]
        t.transform.translation.y = pos[1]
        t.transform.translation.z = pos[2]
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = WorldTFPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
