#!/usr/bin/env python3
"""
MJPEG streamer that subscribes to ROS2 camera topic
Uses the same QoS that worked with ros2 topic echo
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import threading
import time


class MJPEGStreamerNode(Node):
    def __init__(self):
        super().__init__('mjpeg_streamer')
        
        self.bridge = CvBridge()
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.frame_count = 0
        
        # QoS that worked: best_effort + volatile
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            qos
        )
        self.get_logger().info('Subscribed to /camera/image_raw (BEST_EFFORT/VOLATILE)')
        
        # Status timer
        self.timer = self.create_timer(2.0, self.status_callback)
        
    def status_callback(self):
        # self.get_logger().info(f'Frames received: {self.frame_count}')
        pass
        
    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.frame_lock:
                self.current_frame = frame
            self.frame_count += 1
            if self.frame_count == 1:
                self.get_logger().info(f'First frame: {msg.width}x{msg.height}')
        except Exception as e:
            self.get_logger().error(f'Error: {e}')
            
    def get_frame(self):
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None


# Global node reference
ros_node = None


class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = '''<!DOCTYPE html>
<html>
<head>
    <title>Robot Camera MJPEG</title>
    <style>
        body { font-family: Arial; text-align: center; background: #1a1a1a; color: white; padding: 20px; margin: 0; }
        img { max-width: 100%; border: 2px solid #444; border-radius: 8px; }
        h1 { margin-bottom: 20px; }
        .info { color: #888; margin-top: 10px; }
    </style>
</head>
<body>
    <h1>Robot Camera Feed (ROS2)</h1>
    <img src="/stream" alt="Camera Stream">
    <p class="info">MJPEG Stream from /camera/image_raw</p>
</body>
</html>'''
            self.wfile.write(html.encode())
            
        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            
            print(f"Client connected: {self.client_address}")
            
            while True:
                try:
                    frame = ros_node.get_frame() if ros_node else None
                    
                    if frame is None:
                        # Show waiting message
                        frame = np.zeros((480, 640, 3), dtype='uint8')
                        cv2.putText(frame, "Waiting for camera...", (50, 240),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if not ret:
                        continue
                        
                    self.wfile.write(b'--jpgboundary\r\n')
                    self.send_header('Content-type', 'image/jpeg')
                    self.send_header('Content-length', len(jpeg))
                    self.end_headers()
                    self.wfile.write(jpeg.tobytes())
                    self.wfile.write(b'\r\n')
                    
                    time.sleep(0.033)
                    
                except (BrokenPipeError, ConnectionResetError):
                    print(f"Client disconnected: {self.client_address}")
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    break
        else:
            self.send_response(404)
            self.end_headers()
            
    def log_message(self, format, *args):
        pass


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main():
    global ros_node
    
    rclpy.init()
    ros_node = MJPEGStreamerNode()
    
    # ROS2 spin in background
    ros_thread = threading.Thread(target=lambda: rclpy.spin(ros_node), daemon=True)
    ros_thread.start()
    
    # HTTP server
    port = 8081
    server = ThreadedHTTPServer(('0.0.0.0', port), MJPEGHandler)
    print(f"\n{'='*50}")
    print(f"MJPEG Server on http://0.0.0.0:{port}")
    print(f"{'='*50}\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()
        ros_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

