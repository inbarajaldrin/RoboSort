#!/usr/bin/env python3
"""
MJPEG streamer that directly captures from camera using GStreamer
"""

import cv2
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import threading
import time


class CameraCapture:
    def __init__(self):
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.frame_count = 0
        
        # GStreamer pipeline matching jetank_control camera driver
        gst_pipeline = (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=30/1 ! "
            "nvvidconv flip-method=0 ! "
            "video/x-raw, format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! appsink"
        )
        
        self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            print("ERROR: Failed to open camera.")
            exit(1)
        
        print("Camera opened successfully")
        
        # Start capture thread
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("WARNING: Failed to capture frame")
                time.sleep(0.01)
                continue
            
            with self.frame_lock:
                self.current_frame = frame
            self.frame_count += 1
            if self.frame_count == 1:
                h, w = frame.shape[:2]
                print(f'First frame captured: {w}x{h}')
            
    def get_frame(self):
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None
    
    def release(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()


camera = None


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
    <h1>Robot Camera Feed</h1>
    <img src="/stream" alt="Camera Stream">
    <p class="info">MJPEG Stream from Direct Camera Capture</p>
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
                    frame = camera.get_frame() if camera else None
                    
                    if frame is None:
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
    global camera
    
    camera = CameraCapture()
    
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
        if camera:
            camera.release()


if __name__ == '__main__':
    main()


