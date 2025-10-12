#!/usr/bin/env python3
"""
End-Effector Pose Publisher Node
Publishes the gripper center link pose as PoseStamped message
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformListener, Buffer
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
import tf2_geometry_msgs


class EEPosePublisher(Node):
    def __init__(self):
        super().__init__('ee_pose_publisher')
        
        # Declare parameters
        self.declare_parameter('base_frame', 'BEARING_1')
        self.declare_parameter('ee_frame', 'GRIPPER_CENTER_LINK')
        self.declare_parameter('publish_rate', 10.0)  # Hz
        self.declare_parameter('startup_delay', 3.0)  # seconds
        
        # Get parameters
        self.base_frame = self.get_parameter('base_frame').value
        self.ee_frame = self.get_parameter('ee_frame').value
        self.publish_rate = self.get_parameter('publish_rate').value
        startup_delay = self.get_parameter('startup_delay').value
        
        # Create publisher
        self.pose_pub = self.create_publisher(
            PoseStamped, 
            '/ee_pose', 
            10
        )
        
        # TF2 setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.get_logger().info(f'End-Effector Pose Publisher initialized')
        self.get_logger().info(f'Will publish transform from {self.base_frame} to {self.ee_frame}')
        self.get_logger().info(f'Topic: /ee_pose at {self.publish_rate} Hz')
        self.get_logger().info(f'Waiting {startup_delay} seconds before starting...')
        
        # Timer for startup delay - starts publishing after delay
        self.startup_timer = self.create_timer(startup_delay, self.start_publishing)
        self.publish_timer = None
    
    def start_publishing(self):
        """Start the publishing timer after startup delay"""
        self.startup_timer.cancel()
        self.startup_timer.destroy()
        
        # Now create the periodic publishing timer
        self.publish_timer = self.create_timer(1.0 / self.publish_rate, self.publish_ee_pose)
        self.get_logger().info(f'Started publishing end-effector pose')
        
    def publish_ee_pose(self):
        """Get transform and publish as PoseStamped"""
        try:
            # Look up transform
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.ee_frame,
                rclpy.time.Time(),  # Get latest available transform
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
            # Create PoseStamped message
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = self.base_frame
            
            # Fill in position from transform
            pose_msg.pose.position.x = transform.transform.translation.x
            pose_msg.pose.position.y = transform.transform.translation.y
            pose_msg.pose.position.z = transform.transform.translation.z
            
            # Fill in orientation from transform
            pose_msg.pose.orientation.x = transform.transform.rotation.x
            pose_msg.pose.orientation.y = transform.transform.rotation.y
            pose_msg.pose.orientation.z = transform.transform.rotation.z
            pose_msg.pose.orientation.w = transform.transform.rotation.w
            
            # Publish
            self.pose_pub.publish(pose_msg)
            
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            # Only log warning occasionally to avoid spam
            if not hasattr(self, '_last_warning_time'):
                self._last_warning_time = 0
            
            current_time = self.get_clock().now().nanoseconds / 1e9
            if current_time - self._last_warning_time > 5.0:  # Log every 5 seconds
                self.get_logger().warn(
                    f'Could not get transform from {self.base_frame} to {self.ee_frame}: {str(e)}'
                )
                self._last_warning_time = current_time


def main(args=None):
    rclpy.init(args=args)
    node = EEPosePublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

