#!/usr/bin/env python3
"""
Video stream publisher that downloads video from URL and publishes as ROS2 topic
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import time
import sys


class VideoPublisherNode(Node):
    def __init__(self, stream_url):
        super().__init__('video_publisher')
        
        self.stream_url = stream_url
        self.bridge = CvBridge()
        self.frame_count = 0
        self.is_connected = False
        
        # Create publisher for video frames with default QoS
        self.publisher = self.create_publisher(
            Image,
            '/camera/image_rgb',
            10  # Default QoS with queue depth of 10
        )
        self.get_logger().info(f'Publishing to /camera/image_rgb (default QoS)')
        self.get_logger().info(f'Connecting to stream: {stream_url}')
        
        # Initialize video capture
        self.cap = None
        self.connect_to_stream()
        
        # Create timer for reading and publishing frames
        self.frame_timer = self.create_timer(0.033, self.publish_frame)  # ~30 FPS
        
    def connect_to_stream(self):
        """Connect to the video stream URL"""
        try:
            # OpenCV VideoCapture can handle HTTP streams and MJPEG streams
            self.cap = cv2.VideoCapture(self.stream_url)
            
            # Set buffer size to reduce latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not self.cap.isOpened():
                return False
            
            # Try to read a frame to verify connection
            ret, frame = self.cap.read()
            if ret:
                height, width = frame.shape[:2]
                # Only log when reconnecting (state change from disconnected to connected)
                if not self.is_connected:
                    self.get_logger().info(f'Successfully connected! Stream resolution: {width}x{height}')
                self.is_connected = True
                return True
            else:
                return False
                
        except Exception as e:
            return False
    
    def publish_frame(self):
        """Read frame from stream and publish to ROS2 topic"""
        if self.cap is None or not self.cap.isOpened():
            # Log only once when disconnecting
            if self.is_connected:
                self.get_logger().warn('Stream disconnected, attempting to reconnect...')
                self.is_connected = False
            self.connect_to_stream()
            return
        
        try:
            ret, frame = self.cap.read()
            
            if not ret:
                # Log only once when disconnecting
                if self.is_connected:
                    self.get_logger().warn('Stream disconnected, attempting to reconnect...')
                    self.is_connected = False
                # Try to reconnect
                self.cap.release()
                time.sleep(1.0)
                self.connect_to_stream()
                return
            
            # Convert OpenCV frame to ROS2 Image message
            try:
                img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                img_msg.header.stamp = self.get_clock().now().to_msg()
                img_msg.header.frame_id = 'camera_frame'
                
                # Publish the image
                self.publisher.publish(img_msg)
                self.frame_count += 1
                
            except Exception as e:
                self.get_logger().error(f'Error converting/publishing frame: {e}')
                
        except Exception as e:
            # Log only once when disconnecting
            if self.is_connected:
                self.get_logger().warn('Stream disconnected, attempting to reconnect...')
                self.is_connected = False
            # Try to reconnect
            if self.cap is not None:
                self.cap.release()
            time.sleep(1.0)
            self.connect_to_stream()
    
    def status_callback(self):
        """Periodic status update"""
        # if self.cap is not None and self.cap.isOpened():
        #     self.get_logger().info(f'Frames published: {self.frame_count}, Stream active')
        # else:
        #     self.get_logger().warn(f'Frames published: {self.frame_count}, Stream disconnected')
        pass
    
    def destroy_node(self):
        """Clean up resources"""
        if self.cap is not None:
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    # Stream URL - can be modified or passed as argument
    stream_url = 'http://100.88.193.82:8081/stream'
    
    # Check if URL argument is provided
    if len(sys.argv) > 1:
        stream_url = sys.argv[1]
    
    node = VideoPublisherNode(stream_url)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass  


if __name__ == '__main__':
    main()
