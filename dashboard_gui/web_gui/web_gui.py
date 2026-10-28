from flask import Flask, Response, render_template, jsonify
import cv2
import threading
import time
from datetime import datetime, timedelta
import random
import json
import numpy as np
import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
import os

# ROS2 imports
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
    from cv_bridge import CvBridge
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("⚠ ROS2 not available - Camera 2 will use direct capture")

# MQTT imports
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    print("⚠ MQTT not available - install paho-mqtt to enable MQTT support")

app = Flask(__name__)

# Camera variables - Direct camera capture (no network streaming)
camera1 = None
camera2 = None
camera1_id = 6  # Direct USB camera capture (forced)
camera2_id = 1  # Local camera
camera1_frame = None
camera2_frame = None
frame_lock1 = threading.Lock()
frame_lock2 = threading.Lock()
camera1_active = False
camera2_active = False

# Database configuration
DB_CONFIG = {
    'dbname': os.getenv('DB_NAME', 'lego_sorting_db'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', ''),  # Require environment variable for security
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432')
}

# Database connection
db_conn = None
db_lock = threading.Lock()

# ROS2 SQL Update publisher
sql_update_publisher = None
sql_update_node = None

# ROS2 executor for all nodes
ros2_executor = None
ros2_nodes = []

# MQTT configuration
MQTT_CONFIG = {
    'broker': os.getenv('MQTT_BROKER', 'broker.hivemq.com'),  # Match mqtt_publisher.py default
    'port': int(os.getenv('MQTT_PORT', '1883')),
    'topic': os.getenv('MQTT_TOPIC', 'lego_sorting/sql_update'),
    'client_id': os.getenv('MQTT_CLIENT_ID', 'lego_gui_client'),
    'username': os.getenv('MQTT_USERNAME', None),
    'password': os.getenv('MQTT_PASSWORD', None)
}

# MQTT client
mqtt_client = None
mqtt_active = False

color_marker_map = {
    'Red': 1,
    'Blue': 2,
    'Green': 3,
    'Yellow': 4,
    'Orange': 5,
    'Black': 6,
    'White': 7,
    'Purple': 8
}

def create_database():
    """Create PostgreSQL database and tables"""
    global db_conn
    
    try:
        # First connect to default postgres database to create our database
        admin_conn = psycopg2.connect(
            dbname='postgres',
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port']
        )
        admin_conn.autocommit = True
        admin_cursor = admin_conn.cursor()
        
        # Check if database exists, create if not
        admin_cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (DB_CONFIG['dbname'],)
        )
        if not admin_cursor.fetchone():
            admin_cursor.execute(
                sql.SQL("CREATE DATABASE {}").format(
                    sql.Identifier(DB_CONFIG['dbname'])
                )
            )
            print(f"✓ Created database: {DB_CONFIG['dbname']}")
        else:
            print(f"✓ Database already exists: {DB_CONFIG['dbname']}")
        
        admin_cursor.close()
        admin_conn.close()
        
        # Connect to our database
        db_conn = psycopg2.connect(**DB_CONFIG)
        db_conn.autocommit = True
        cursor = db_conn.cursor()
        
        # Create sorting_records table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sorting_records (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                color VARCHAR(50) NOT NULL,
                aruco_marker_id INTEGER NOT NULL,
                count INTEGER NOT NULL,
                status VARCHAR(50) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index on timestamp for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON sorting_records(timestamp DESC)
        """)
        
        # Create index on status
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_status 
            ON sorting_records(status)
        """)
        
        cursor.close()
        print("✓ Database tables created successfully")
        return True
        
    except psycopg2.Error as e:
        print(f"⚠ Database error: {e}")
        print("⚠ Falling back to in-memory data storage")
        return False
    except Exception as e:
        print(f"⚠ Database connection error: {e}")
        print("⚠ Falling back to in-memory data storage")
        return False

def clear_database():
    """Clear all entries from the database"""
    global db_conn
    
    if not db_conn:
        return False
    
    try:
        cursor = db_conn.cursor()
        
        # Delete all records
        cursor.execute("DELETE FROM sorting_records")
        deleted_count = cursor.rowcount
        
        db_conn.commit()
        cursor.close()
        
        print(f"✓ Cleared database: Removed {deleted_count} records")
        return True
        
    except psycopg2.Error as e:
        print(f"⚠ Error clearing database: {e}")
        return False

def initialize_fake_data():
    """Initialize database (clears existing data)"""
    global db_conn
    
    if not db_conn:
        return
    
    try:
        # Clear all existing data
        clear_database()
        print("✓ Database initialized (empty - waiting for /sql_update messages)")
        
    except Exception as e:
        print(f"⚠ Error initializing database: {e}")

def insert_sorting_record(color, aruco_id, count, status='Processing', timestamp=None):
    """Insert a new sorting record into the database"""
    global db_conn
    
    if not db_conn:
        return None
    
    # Ensure status is only Processing or Completed
    if status not in ['Processing', 'Completed']:
        status = 'Processing'
    
    # IMPORTANT: For single robot workflow - only ONE Processing record should exist at a time
    # If inserting Processing, check if there's already a Processing record (any color/ArUco)
    # If so, that means the previous object wasn't completed - update it first or skip
    if status == 'Processing':
        try:
            cursor = db_conn.cursor()
            # Check if there's ANY Processing record (single robot = only one at a time)
            cursor.execute("""
                SELECT id, aruco_marker_id, color FROM sorting_records
                WHERE status = 'Processing'
                ORDER BY timestamp DESC, id DESC
                LIMIT 1
            """)
            
            existing = cursor.fetchone()
            if existing:
                existing_id, existing_aruco, existing_color = existing
                print(f"⚠ Another Processing record exists (ID: {existing_id}, ArUco: {existing_aruco}, Color: {existing_color})")
                print(f"   Skipping new Processing insert. Complete the current object first or use update action.")
                cursor.close()
                return None
            cursor.close()
        except Exception as e:
            print(f"⚠ Error checking for existing Processing record: {e}")
    
    try:
        cursor = db_conn.cursor()
        
        # Use provided timestamp or current timestamp
        if timestamp:
            cursor.execute("""
                INSERT INTO sorting_records (timestamp, color, aruco_marker_id, count, status)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id, timestamp, color, aruco_marker_id, count, status
            """, (timestamp, color, aruco_id, count, status))
        else:
            cursor.execute("""
                INSERT INTO sorting_records (timestamp, color, aruco_marker_id, count, status)
                VALUES (CURRENT_TIMESTAMP, %s, %s, %s, %s)
                RETURNING id, timestamp, color, aruco_marker_id, count, status
            """, (color, aruco_id, count, status))
        
        record = cursor.fetchone()
        db_conn.commit()
        cursor.close()
        
        result = {
            'id': record[0],
            'timestamp': record[1].strftime('%Y-%m-%d %H:%M:%S'),
            'color': record[2],
            'aruco_marker_id': record[3],
            'count': record[4],
            'status': record[5]
        }
        
        print(f"✓ Inserted record: ID {result['id']}, ArUco {aruco_id}, Color {color}, Status {status}")
        
        return result
    except psycopg2.Error as e:
        print(f"⚠ Error inserting record: {e}")
        return None

def update_record_status(record_id: int, new_status: str) -> bool:
    """Update the status of a record by ID"""
    global db_conn, sql_update_publisher
    
    if not db_conn:
        return False
    
    if new_status not in ['Processing', 'Completed']:
        return False
    
    try:
        cursor = db_conn.cursor()
        cursor.execute("""
            UPDATE sorting_records
            SET status = %s
            WHERE id = %s
            RETURNING id, timestamp, color, aruco_marker_id, count, status
        """, (new_status, record_id))
        
        record = cursor.fetchone()
        db_conn.commit()
        updated = cursor.rowcount > 0
        cursor.close()
        
        # Publish update to /sql_update topic
        if updated and ROS2_AVAILABLE and sql_update_publisher:
            try:
                result = {
                    'id': record[0],
                    'timestamp': record[1].strftime('%Y-%m-%d %H:%M:%S'),
                    'color': record[2],
                    'aruco_marker_id': record[3],
                    'count': record[4],
                    'status': record[5]
                }
                publish_sql_update(result)
            except Exception as e:
                print(f"⚠ Error publishing update to /sql_update: {e}")
        
        return updated
    except psycopg2.Error as e:
        print(f"⚠ Error updating record status: {e}")
        return False

def update_processing_to_completed(aruco_marker_id: int, color: str = None) -> bool:
    """Update the most recent Processing record to Completed for a given ArUco marker ID"""
    global db_conn
    
    if not db_conn:
        return False
    
    try:
        cursor = db_conn.cursor()
        
        # First, find the most recent Processing record
        if color:
            cursor.execute("""
                SELECT id FROM sorting_records
                WHERE aruco_marker_id = %s 
                AND color = %s
                AND status = 'Processing'
                ORDER BY timestamp DESC, id DESC
                LIMIT 1
            """, (aruco_marker_id, color))
        else:
            cursor.execute("""
                SELECT id FROM sorting_records
                WHERE aruco_marker_id = %s
                AND status = 'Processing'
                ORDER BY timestamp DESC, id DESC
                LIMIT 1
            """, (aruco_marker_id,))
        
        record_id = cursor.fetchone()
        
        if not record_id:
            cursor.close()
            print(f"⚠ No Processing record found to update (ArUco ID: {aruco_marker_id}, Color: {color or 'any'})")
            return False
        
        record_id = record_id[0]
        
        # Update the found record
        cursor.execute("""
            UPDATE sorting_records
            SET status = 'Completed'
            WHERE id = %s
            AND status = 'Processing'
            RETURNING id, timestamp, color, aruco_marker_id, count, status
        """, (record_id,))
        
        record = cursor.fetchone()
        db_conn.commit()
        updated = cursor.rowcount > 0
        cursor.close()
        
        if updated:
            print(f"✓ Updated record ID {record[0]} from Processing to Completed (ArUco ID: {aruco_marker_id}, Color: {color or 'any'})")
        else:
            print(f"⚠ Record ID {record_id} was not in Processing status (may have been updated already)")
        
        return updated
    except psycopg2.Error as e:
        print(f"⚠ Error updating Processing to Completed: {e}")
        return False

def get_recent_records(limit=20):
    """Get recent sorting records from database (always fresh query)"""
    global db_conn
    
    if not db_conn:
        return []
    
    try:
        # Always execute fresh query (no caching)
        cursor = db_conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT id, timestamp, color, aruco_marker_id, count, status
            FROM sorting_records
            ORDER BY timestamp DESC
            LIMIT %s
        """, (limit,))
        
        records = cursor.fetchall()
        cursor.close()
        
        # Convert to list of dicts with formatted timestamp
        result = []
        for record in records:
            result.append({
                'id': record['id'],
                'timestamp': record['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'color': record['color'],
                'aruco_marker_id': record['aruco_marker_id'],
                'count': record['count'],
                'status': record['status']
            })
        
        return result
    except psycopg2.Error as e:
        print(f"⚠ Error fetching records from database: {e}")
        return []
    except Exception as e:
        print(f"⚠ Unexpected error fetching records: {e}")
        return []

def get_statistics():
    """Get database statistics"""
    global db_conn
    
    if not db_conn:
        return {'total': 0, 'processing': 0, 'completed': 0}
    
    try:
        cursor = db_conn.cursor()
        
        # Total count of LEGO pieces sorted (sum of all counts)
        cursor.execute("SELECT SUM(count) FROM sorting_records")
        total_result = cursor.fetchone()[0]
        total = int(total_result) if total_result else 0
        
        # Processing count of LEGO pieces (sum of counts for Processing records)
        cursor.execute("SELECT SUM(count) FROM sorting_records WHERE status = 'Processing'")
        processing_result = cursor.fetchone()[0]
        processing = int(processing_result) if processing_result else 0
        
        # Completed count of LEGO pieces (sum of counts for Completed records)
        cursor.execute("SELECT SUM(count) FROM sorting_records WHERE status = 'Completed'")
        completed_result = cursor.fetchone()[0]
        completed = int(completed_result) if completed_result else 0
        
        cursor.close()
        
        return {
            'total': total,
            'processing': processing,
            'completed': completed
        }
    except psycopg2.Error as e:
        print(f"⚠ Error fetching statistics: {e}")
        return {'total': 0, 'processing': 0, 'completed': 0}

def get_graph_data():
    """Get graph data for bar chart: color vs total count with ArUco marker IDs"""
    global db_conn
    
    if not db_conn:
        return {'labels': [], 'datasets': [], 'aruco_ids': {}}
    
    try:
        cursor = db_conn.cursor()
        
        # Get total count per color and their ArUco marker IDs
        cursor.execute("""
            SELECT 
                color,
                SUM(count) as total_count,
                MAX(aruco_marker_id) as aruco_id
            FROM sorting_records
            GROUP BY color
            ORDER BY color
        """)
        
        records = cursor.fetchall()
        cursor.close()
        
        # Build data for bar chart
        colors = []
        counts = []
        aruco_ids = {}
        
        color_map = {
            'Red': 'rgb(255, 0, 0)',
            'Blue': 'rgb(0, 0, 255)',
            'Green': 'rgb(0, 255, 0)',
            'Yellow': 'rgb(255, 255, 0)',
            'Orange': 'rgb(255, 165, 0)',
            'Black': 'rgb(0, 0, 0)',
            'White': 'rgb(255, 255, 255)',
            'Purple': 'rgb(128, 0, 128)'
        }
        
        background_colors = []
        border_colors = []
        
        for record in records:
            color, total_count, aruco_id = record
            colors.append(color)
            # Ensure count is an integer
            count_value = int(total_count) if total_count else 0
            counts.append(count_value)
            aruco_ids[color] = int(aruco_id)
            background_colors.append(color_map.get(color, 'rgb(128, 128, 128)'))
            border_colors.append(color_map.get(color, 'rgb(128, 128, 128)'))
        
        # If no data, show all colors with 0 count
        if len(colors) == 0:
            for color in color_marker_map.keys():
                colors.append(color)
                counts.append(0)
                aruco_ids[color] = color_marker_map[color]
                background_colors.append(color_map.get(color, 'rgb(128, 128, 128)'))
                border_colors.append(color_map.get(color, 'rgb(128, 128, 128)'))
        
        dataset = {
            'label': 'Total Count',
            'data': counts,
            'backgroundColor': background_colors,
            'borderColor': border_colors,
            'borderWidth': 2
        }
        
        return {
            'labels': colors,
            'datasets': [dataset],
            'aruco_ids': aruco_ids
        }
    except psycopg2.Error as e:
        print(f"⚠ Error fetching graph data: {e}")
        return {'labels': [], 'datasets': [], 'aruco_ids': {}}

# Fake data generator removed - database updates now come only from /sql_update ROS2 topic

def generate_camera_frame(camera, frame_lock, frame_var):
    """Generate frames from camera"""
    global camera1_active, camera2_active
    active_var = camera1_active if camera == camera1 else camera2_active
    
    while active_var:
        if camera and camera.isOpened():
            ret, frame = camera.read()
            if ret:
                # Reduced resolution for better performance (320x240)
                frame = cv2.resize(frame, (320, 240))
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    with frame_lock:
                        frame_var = buffer.tobytes()
        time.sleep(0.05)  # ~20 FPS for better performance

def detect_available_cameras(max_cams=20):
    """Detect available cameras"""
    available = []
    print("Scanning for available cameras...")
    
    # Check /dev/video* devices
    try:
        import glob
        video_devices = sorted(glob.glob('/dev/video*'))
        if video_devices:
            print(f"Found video devices: {video_devices}")
            for dev in video_devices:
                try:
                    dev_num = int(dev.replace('/dev/video', ''))
                    if dev_num < max_cams:
                        cap = cv2.VideoCapture(dev_num, cv2.CAP_V4L2)
                        if cap.isOpened():
                            ret, _ = cap.read()
                            if ret:
                                available.append(dev_num)
                                print(f"  ✓ Camera {dev_num} at {dev}")
                            cap.release()
                except:
                    continue
    except Exception as e:
        print(f"  Note: {e}")
    
    # Also check standard indices
    for i in range(max_cams):
        if i not in available:
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        available.append(i)
                        print(f"  ✓ Camera {i}")
                    cap.release()
            except:
                continue
    
    return sorted(list(set(available)))

def check_camera_permissions(device_path):
    """Check if user has permission to access camera device"""
    import os
    import stat
    
    if not os.path.exists(device_path):
        return False, "Device does not exist"
    
    # Check if readable
    if not os.access(device_path, os.R_OK):
        return False, "Permission denied - user not in video group"
    
    return True, "OK"

def camera1_thread():
    """Camera 1 capture thread - Direct USB camera capture (no network)"""
    global camera1, camera1_frame, camera1_active
    camera1_active = True
    
    # First, detect available cameras
    available_cameras = detect_available_cameras()
    
    if available_cameras:
        print(f"Available cameras: {available_cameras}")
    else:
        print("⚠ No cameras detected")
    
    # Check if requested camera ID is available
    if camera1_id not in available_cameras and available_cameras:
        print(f"⚠ Camera ID {camera1_id} not found. Available cameras: {available_cameras}")
        print(f"   Trying to use camera ID {camera1_id} anyway...")
    
    try:
        print(f"Opening Camera 1 (ID: {camera1_id})...")
        
        # Check permissions first
        device_path = f"/dev/video{camera1_id}"
        import os
        if os.path.exists(device_path):
            has_permission, msg = check_camera_permissions(device_path)
            if not has_permission:
                print(f"✗ Permission error: {msg}")
                print(f"   Fix: sudo usermod -a -G video $USER")
                print(f"   Then log out and log back in")
                camera1 = None
                return
        
        # Try multiple methods to open the camera
        camera1 = None
        
        # Method 1: Try V4L2 backend with device path
        try:
            if os.path.exists(device_path):
                print(f"  Trying {device_path} with V4L2...")
                camera1 = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
                if camera1.isOpened():
                    ret, _ = camera1.read()
                    if ret:
                        print(f"  ✓ Opened {device_path} successfully")
                    else:
                        camera1.release()
                        camera1 = None
        except Exception as e:
            print(f"  Method 1 failed: {e}")
        
        # Method 2: Try V4L2 backend with index
        if not camera1 or not camera1.isOpened():
            try:
                print(f"  Trying camera index {camera1_id} with V4L2...")
                camera1 = cv2.VideoCapture(camera1_id, cv2.CAP_V4L2)
                if camera1.isOpened():
                    ret, _ = camera1.read()
                    if ret:
                        print(f"  ✓ Opened camera {camera1_id} with V4L2")
                    else:
                        camera1.release()
                        camera1 = None
            except Exception as e:
                print(f"  Method 2 failed: {e}")
        
        # Method 3: Try default backend
        if not camera1 or not camera1.isOpened():
            try:
                print(f"  Trying camera index {camera1_id} with default backend...")
                camera1 = cv2.VideoCapture(camera1_id)
                if camera1.isOpened():
                    ret, _ = camera1.read()
                    if ret:
                        print(f"  ✓ Opened camera {camera1_id} with default backend")
                    else:
                        camera1.release()
                        camera1 = None
            except Exception as e:
                print(f"  Method 3 failed: {e}")
        
        # Method 4: Try first available camera if requested one fails
        if (not camera1 or not camera1.isOpened()) and available_cameras:
            try:
                fallback_id = available_cameras[0]
                print(f"  ⚠ Camera {camera1_id} failed, trying fallback camera {fallback_id}...")
                camera1 = cv2.VideoCapture(fallback_id, cv2.CAP_V4L2)
                if not camera1.isOpened():
                    camera1 = cv2.VideoCapture(fallback_id)
                if camera1.isOpened():
                    ret, _ = camera1.read()
                    if ret:
                        print(f"  ✓ Using fallback camera {fallback_id}")
                    else:
                        camera1.release()
                        camera1 = None
            except Exception as e:
                print(f"  Fallback failed: {e}")
        
        if not camera1 or not camera1.isOpened():
            print(f"✗ Failed to open Camera 1 (ID: {camera1_id})")
            if available_cameras:
                print(f"   Available cameras: {available_cameras}")
                print(f"   Try changing camera1_id in web_gui.py to one of these IDs")
            print(f"   Permission fix: sudo usermod -a -G video $USER && newgrp video")
            camera1 = None
            return
        
        # Optimize camera settings for better quality and higher FPS
        try:
            camera1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera1.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Single buffer for low latency
            camera1.set(cv2.CAP_PROP_FPS, 25)  # Higher FPS for smoother video
        except Exception as e:
            print(f"  Note: Some camera properties could not be set: {e}")
        
        # Verify camera is actually working
        ret, test_frame = camera1.read()
        if not ret or test_frame is None:
            print(f"✗ Camera opened but cannot read frames")
            camera1.release()
            camera1 = None
            return
        
        print(f"✓ Camera 1 opened successfully (ID: {camera1_id})")
        print(f"  Frame size: {test_frame.shape[1]}x{test_frame.shape[0]}")
        
        while camera1_active:
            ret, frame = camera1.read()
            if ret and frame is not None:
                # Resize to medium resolution (balance between quality and performance)
                frame = cv2.resize(frame, (480, 360), interpolation=cv2.INTER_LINEAR)
                
                # Encode as JPEG with balanced quality
                ret, buffer = cv2.imencode('.jpg', frame, [
                    cv2.IMWRITE_JPEG_QUALITY, 65,
                    cv2.IMWRITE_JPEG_OPTIMIZE, 1
                ])
                if ret:
                    with frame_lock1:
                        camera1_frame = buffer.tobytes()
            else:
                break
            
            time.sleep(0.04)  # ~25 FPS for smoother video
            
    except Exception as e:
        print(f"Camera 1 error: {e}")
    finally:
        if camera1:
            try:
                camera1.release()
            except:
                pass
        camera1 = None
        camera1_active = False
        print("Camera 1 thread stopped")

class ROS2ImageSubscriber(Node):
    """ROS2 node to subscribe to /camera_annotated topic"""
    def __init__(self):
        super().__init__('camera_annotated_subscriber')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/camera_annotated',
            self.image_callback,
            10)
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
    def image_callback(self, msg):
        try:
            # Convert ROS2 Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.frame_lock:
                self.latest_frame = cv_image
        except Exception as e:
            # Silently handle conversion errors to avoid spam
            if hasattr(self, 'get_logger'):
                try:
                    self.get_logger().error(f'Error converting image: {e}')
                except:
                    pass

def process_sql_update_message(message_data):
    """Process SQL update message (shared by ROS2 and MQTT)"""
    try:
        # Parse JSON message if it's a string
        if isinstance(message_data, str):
            data = json.loads(message_data)
        else:
            data = message_data
        
        # Check if this is an update action or insert action
        action = data.get('action', 'insert')  # Default to insert for backward compatibility
        
        if action == 'update':
            # Update existing Processing record to Completed
            aruco_id = int(data.get('aruco_marker_id'))
            color = data.get('color')
            
            if not aruco_id:
                print('⚠ Missing aruco_marker_id for update action')
                return False
            
            # Update the most recent Processing record
            updated = update_processing_to_completed(aruco_id, color)
            
            if updated:
                print(f'✓ Updated Processing to Completed for ArUco ID {aruco_id}, Color {color or "any"}')
            else:
                print(f'⚠ No Processing record found to update for ArUco ID {aruco_id}, Color {color or "any"}')
            return updated
        
        elif action == 'update_by_id':
            # Update specific record by ID
            record_id = int(data.get('record_id'))
            new_status = str(data.get('status', 'Completed'))
            
            if not record_id:
                print('⚠ Missing record_id for update_by_id action')
                return False
            
            updated = update_record_status(record_id, new_status)
            
            if updated:
                print(f'✓ Successfully updated record ID {record_id} to {new_status}')
            else:
                print(f'⚠ Failed to update record ID {record_id}')
            return updated
        
        else:
            # Default: Insert new record (or update if Processing exists and status is Completed)
            # Validate required fields
            required_fields = ['aruco_marker_id', 'color', 'status', 'count']
            if not all(field in data for field in required_fields):
                print(f'⚠ Missing required fields in SQL update message: {data}')
                return False
            
            # Extract data
            aruco_id = int(data['aruco_marker_id'])
            color = str(data['color'])
            status = str(data['status'])
            count = int(data['count'])
            timestamp = None
            
            # Parse timestamp if provided
            if 'timestamp' in data:
                try:
                    timestamp = datetime.strptime(data['timestamp'], '%Y-%m-%d %H:%M:%S')
                except:
                    # Try ISO format
                    try:
                        timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
                    except:
                        pass
            
            # Smart insert/update logic:
            # If status is "Completed", ALWAYS update existing Processing record (single robot workflow)
            if status == 'Completed':
                # Try to find and update the Processing record
                # For single robot, there should be only one Processing record
                updated = update_processing_to_completed(aruco_id, color)
                if updated:
                    print(f'✓ Updated Processing record to Completed for ArUco ID {aruco_id}, Color {color}')
                    return True
                else:
                    # If no Processing record found, don't create a Completed record
                    print(f'⚠ No Processing record found for ArUco ID {aruco_id}, Color {color}. Cannot complete without Processing first.')
                    return False  # Don't insert Completed without Processing
            
            # Insert into database
            result = insert_sorting_record(
                color=color,
                aruco_id=aruco_id,
                count=count,
                status=status,
                timestamp=timestamp
            )
            
            if result:
                print(f'✓ Successfully inserted record: ID {result["id"]}')
                return True
            else:
                print(f'⚠ Failed to insert record')
                return False
                
    except json.JSONDecodeError as e:
        print(f'⚠ Invalid JSON in SQL update message: {e}')
        return False
    except Exception as e:
        print(f'⚠ Error processing SQL update message: {e}')
        return False

class SQLUpdateSubscriber(Node):
    """ROS2 node to subscribe to /sql_update topic and update database"""
    def __init__(self):
        super().__init__('sql_update_subscriber')
        self.subscription = self.create_subscription(
            String,
            '/sql_update',
            self.sql_update_callback,
            10)
        
    def sql_update_callback(self, msg):
        """Process SQL update message from /sql_update topic"""
        try:
            # Parse JSON message and process using shared function
            data = json.loads(msg.data)
            process_sql_update_message(data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f'Invalid JSON in /sql_update message: {e}')
        except Exception as e:
            self.get_logger().error(f'Error processing /sql_update message: {e}')

def publish_sql_update(record_data):
    """Publish database update to /sql_update ROS2 topic"""
    global sql_update_publisher
    
    if not ROS2_AVAILABLE or not sql_update_publisher:
        return
    
    try:
        # Format message as JSON
        message_data = {
            'aruco_marker_id': record_data['aruco_marker_id'],
            'color': record_data['color'],
            'status': record_data['status'],
            'count': record_data['count'],
            'timestamp': record_data['timestamp'],
            'id': record_data['id']
        }
        
        msg = String()
        msg.data = json.dumps(message_data)
        sql_update_publisher.publish(msg)
        
    except Exception as e:
        print(f"⚠ Error publishing SQL update: {e}")

def init_sql_update_publisher():
    """Initialize ROS2 publisher and subscriber for /sql_update topic"""
    global sql_update_publisher, sql_update_node, ros2_executor, ros2_nodes
    
    if not ROS2_AVAILABLE:
        return
    
    try:
        if not rclpy.ok():
            rclpy.init()
        
        # Create node for SQL update
        sql_update_node = SQLUpdateSubscriber()
        ros2_nodes.append(sql_update_node)
        
        # Create publisher
        sql_update_publisher = sql_update_node.create_publisher(String, '/sql_update', 10)
        
        print("✓ ROS2 /sql_update publisher and subscriber initialized")
        
    except Exception as e:
        print(f"⚠ Error initializing SQL update publisher: {e}")
        sql_update_publisher = None
        sql_update_node = None

# MQTT callback functions (using VERSION2 API for better compatibility)
def on_mqtt_connect(client, userdata, flags, rc, properties=None):
    """Callback when MQTT client connects"""
    if rc == 0:
        print(f"✓ MQTT connected to broker {MQTT_CONFIG['broker']}:{MQTT_CONFIG['port']}")
        # Subscribe to the topic
        result, mid = client.subscribe(MQTT_CONFIG['topic'], qos=1)
        if result == mqtt.MQTT_ERR_SUCCESS:
            print(f"✓ MQTT subscribed to topic: {MQTT_CONFIG['topic']} (mid={mid})")
        else:
            print(f"⚠ MQTT subscription failed (result={result})")
    else:
        print(f"⚠ MQTT connection failed with code {rc}")

def on_mqtt_disconnect(client, userdata, rc, properties=None):
    """Callback when MQTT client disconnects"""
    if rc != 0:
        print(f"⚠ MQTT unexpected disconnection (code {rc})")
    else:
        print("✓ MQTT disconnected")

def on_mqtt_message(client, userdata, msg):
    """Callback when MQTT message is received"""
    try:
        # Decode message payload
        message_str = msg.payload.decode('utf-8')
        print(f"📨 MQTT message received on {msg.topic} (QoS={msg.qos}, retain={msg.retain})")
        print(f"   payload: {message_str[:100]}{'...' if len(message_str) > 100 else ''}")
        
        # Process the message using shared function
        process_sql_update_message(message_str)
    except Exception as e:
        print(f"⚠ Error processing MQTT message: {e}")

def on_mqtt_subscribe(client, userdata, mid, granted_qos, properties=None):
    """Callback when MQTT subscription is confirmed"""
    print(f"✓ MQTT subscription confirmed (mid={mid}, QoS: {granted_qos})")

def init_mqtt_client():
    """Initialize MQTT client and connect to broker"""
    global mqtt_client, mqtt_active
    
    if not MQTT_AVAILABLE:
        print("⚠ MQTT not available - install paho-mqtt to enable MQTT support")
        return
    
    try:
        # Create MQTT client with VERSION2 callback API (for better compatibility)
        try:
            mqtt_client = mqtt.Client(
                client_id=MQTT_CONFIG['client_id'],
                clean_session=True,
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2
            )
        except AttributeError:
            # Fallback to VERSION1 if VERSION2 not available (older paho-mqtt)
            mqtt_client = mqtt.Client(
                client_id=MQTT_CONFIG['client_id'],
                clean_session=True
            )
            # Use VERSION1 callbacks
            def on_connect_v1(client, userdata, flags, rc):
                on_mqtt_connect(client, userdata, flags, rc, None)
            def on_disconnect_v1(client, userdata, rc):
                on_mqtt_disconnect(client, userdata, rc, None)
            def on_subscribe_v1(client, userdata, mid, granted_qos):
                on_mqtt_subscribe(client, userdata, mid, granted_qos, None)
            mqtt_client.on_connect = on_connect_v1
            mqtt_client.on_disconnect = on_disconnect_v1
            mqtt_client.on_subscribe = on_subscribe_v1
        else:
            # Set VERSION2 callbacks
            mqtt_client.on_connect = on_mqtt_connect
            mqtt_client.on_disconnect = on_mqtt_disconnect
            mqtt_client.on_subscribe = on_mqtt_subscribe
        
        # Message callback is the same for both versions
        mqtt_client.on_message = on_mqtt_message
        
        # Set username/password if provided
        if MQTT_CONFIG['username'] and MQTT_CONFIG['password']:
            mqtt_client.username_pw_set(MQTT_CONFIG['username'], MQTT_CONFIG['password'])
        
        # Connect to broker
        print(f"Connecting to MQTT broker {MQTT_CONFIG['broker']}:{MQTT_CONFIG['port']}...")
        mqtt_client.connect(
            MQTT_CONFIG['broker'],
            MQTT_CONFIG['port'],
            keepalive=60
        )
        
        # Start MQTT loop in a separate thread
        def mqtt_loop():
            global mqtt_active
            mqtt_active = True
            try:
                mqtt_client.loop_forever()
            except Exception as e:
                print(f"⚠ MQTT loop error: {e}")
            finally:
                mqtt_active = False
        
        threading.Thread(target=mqtt_loop, daemon=True).start()
        print("✓ MQTT client initialized and running")
        
    except Exception as e:
        print(f"⚠ Error initializing MQTT client: {e}")
        mqtt_client = None

def start_ros2_executor():
    """Start ROS2 executor in a single thread for all nodes"""
    global ros2_executor, ros2_nodes
    
    if not ROS2_AVAILABLE or ros2_executor is not None:
        return
    
    if len(ros2_nodes) == 0:
        return  # No nodes to add yet
    
    try:
        from rclpy.executors import SingleThreadedExecutor
        
        ros2_executor = SingleThreadedExecutor()
        
        # Add all nodes to executor
        for node in ros2_nodes:
            if node is not None:
                ros2_executor.add_node(node)
        
        # Start executor in separate thread
        def executor_spin():
            try:
                while rclpy.ok():
                    ros2_executor.spin_once(timeout_sec=0.1)
            except Exception as e:
                if "generator already executing" not in str(e):
                    print(f"⚠ ROS2 executor error: {e}")
        
        threading.Thread(target=executor_spin, daemon=True).start()
        print(f"✓ ROS2 executor started for {len(ros2_nodes)} node(s)")
        
    except Exception as e:
        print(f"⚠ Error starting ROS2 executor: {e}")
        ros2_executor = None

def camera2_thread():
    """Camera 2 capture thread - ROS2 topic /camera_annotated"""
    global camera2_frame, camera2_active
    
    camera2_active = True
    ros2_node = None
    
    if ROS2_AVAILABLE:
        try:
            print("Initializing ROS2 for Camera 2 (/camera_annotated topic)...")
            if not rclpy.ok():
                rclpy.init()
            
            ros2_node = ROS2ImageSubscriber()
            ros2_nodes.append(ros2_node)
            print("✓ ROS2 subscriber created for /camera_annotated")
            
            # Start ROS2 executor if not already started
            if ros2_executor is None:
                start_ros2_executor()
            else:
                # Add new node to existing executor
                try:
                    ros2_executor.add_node(ros2_node)
                except:
                    pass
            
            # Wait a moment for first message
            time.sleep(1)
            
            print("Camera 2: Listening to /camera_annotated ROS2 topic")
            
            while camera2_active:
                if ros2_node and ros2_node.latest_frame is not None:
                    with ros2_node.frame_lock:
                        frame = ros2_node.latest_frame.copy()
                    
                    if frame is not None:
                        # Resize to medium resolution
                        frame = cv2.resize(frame, (480, 360), interpolation=cv2.INTER_LINEAR)
                        
                        # Encode as JPEG
                        ret, buffer = cv2.imencode('.jpg', frame, [
                            cv2.IMWRITE_JPEG_QUALITY, 65,
                            cv2.IMWRITE_JPEG_OPTIMIZE, 1
                        ])
                        if ret:
                            with frame_lock2:
                                camera2_frame = buffer.tobytes()
                
                time.sleep(0.04)  # ~25 FPS
                
        except Exception as e:
            print(f"ROS2 Camera 2 error: {e}")
            print("Falling back to direct camera capture...")
            ros2_node = None
    
    # Fallback to direct camera capture if ROS2 fails
    if not ros2_node:
        try:
            print(f"Opening Camera 2 directly (ID: {camera2_id})...")
            camera2 = cv2.VideoCapture(camera2_id, cv2.CAP_V4L2)
            if not camera2.isOpened():
                camera2 = cv2.VideoCapture(camera2_id)
            
            if not camera2.isOpened():
                print(f"Failed to open Camera 2 (ID: {camera2_id})")
                return
            
            camera2.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            camera2.set(cv2.CAP_PROP_FPS, 25)
            
            print(f"Camera 2 opened successfully (ID: {camera2_id})")
            
            while camera2_active:
                ret, frame = camera2.read()
                if ret and frame is not None:
                    frame = cv2.resize(frame, (480, 360), interpolation=cv2.INTER_LINEAR)
                    ret, buffer = cv2.imencode('.jpg', frame, [
                        cv2.IMWRITE_JPEG_QUALITY, 65,
                        cv2.IMWRITE_JPEG_OPTIMIZE, 1
                    ])
                    if ret:
                        with frame_lock2:
                            camera2_frame = buffer.tobytes()
                else:
                    break
                
                time.sleep(0.04)  # ~25 FPS
                
            if camera2:
                camera2.release()
        except Exception as e:
            print(f"Camera 2 error: {e}")
    
    camera2_active = False
    print("Camera 2 thread stopped")

def generate_stream(camera_frame, frame_lock):
    """Generate video stream"""
    # Create placeholder image once (reduced resolution)
    placeholder_img = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.putText(placeholder_img, 'Camera Not Available', (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    _, placeholder_buffer = cv2.imencode('.jpg', placeholder_img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    placeholder_bytes = placeholder_buffer.tobytes()
    
    while True:
        with frame_lock:
            if camera_frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + camera_frame + b'\r\n')
            else:
                # Send placeholder
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + placeholder_bytes + b'\r\n')
        time.sleep(0.05)  # ~20 FPS for better performance

@app.route('/')
def index():
    """Main page"""
    import time
    return render_template('index.html', timestamp=int(time.time()))

@app.route('/video_feed1')
def video_feed1():
    """Camera 1 video stream (legacy MJPEG)"""
    return Response(generate_stream(camera1_frame, frame_lock1),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    """Camera 2 video stream (legacy MJPEG)"""
    return Response(generate_stream(camera2_frame, frame_lock2),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/frame1/raw')
def get_frame1_raw():
    """Raw frame endpoint for Camera 1 - WebCodecs"""
    with frame_lock1:
        if camera1_frame is not None:
            return Response(
                camera1_frame,
                mimetype='image/jpeg',
                headers={
                    'Cache-Control': 'no-cache, no-store, must-revalidate',
                    'Pragma': 'no-cache',
                    'Expires': '0'
                }
            )
    return Response(status=204)

@app.route('/api/frame2/raw')
def get_frame2_raw():
    """Raw frame endpoint for Camera 2 - WebCodecs"""
    with frame_lock2:
        if camera2_frame is not None:
            return Response(
                camera2_frame,
                mimetype='image/jpeg',
                headers={
                    'Cache-Control': 'no-cache, no-store, must-revalidate',
                    'Pragma': 'no-cache',
                    'Expires': '0'
                }
            )
    return Response(status=204)

@app.route('/api/database')
def get_database():
    """Get database records from PostgreSQL (always fresh data)"""
    # Always fetch fresh data from database (no caching)
    records = get_recent_records(limit=20)
    response = jsonify(records)
    # Prevent caching to ensure fresh data
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/api/statistics')
def api_statistics():
    """Get statistics from PostgreSQL (always fresh data)"""
    # Always fetch fresh data from database (no caching)
    stats = get_statistics()
    response = jsonify(stats)
    # Prevent caching to ensure fresh data
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/api/graph_data')
def api_graph_data():
    """Get graph data from PostgreSQL (always fresh data)"""
    # Always fetch fresh data from database (no caching)
    graph_data = get_graph_data()
    response = jsonify(graph_data)
    # Prevent caching to ensure fresh data
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

if __name__ == '__main__':
    import numpy as np
    
    print("LEGO Sorting System Web GUI")
    print("=" * 50)
    print(f"Camera 1: Direct capture from USB camera ID {camera1_id}")
    if ROS2_AVAILABLE:
        print(f"Camera 2: ROS2 topic /camera_annotated")
    else:
        print(f"Camera 2: Direct capture from camera ID {camera2_id} (ROS2 not available)")
    print("=" * 50)
    
    # Initialize ROS2 if available
    if ROS2_AVAILABLE:
        try:
            if not rclpy.ok():
                rclpy.init()
            print("✓ ROS2 initialized")
            
            # Initialize SQL update publisher/subscriber
            init_sql_update_publisher()
            
            # Start ROS2 executor for all nodes (will add camera2 node when it's created)
            if len(ros2_nodes) > 0:
                start_ros2_executor()
        except Exception as e:
            print(f"⚠ ROS2 initialization failed: {e}")
    
    # Initialize MQTT if available
    if MQTT_AVAILABLE:
        try:
            init_mqtt_client()
        except Exception as e:
            print(f"⚠ MQTT initialization failed: {e}")
    
    # Initialize PostgreSQL database
    print("\nInitializing PostgreSQL database...")
    if create_database():
        initialize_fake_data()
        print("✓ Database ready - waiting for SQL update messages (ROS2 and/or MQTT)")
    else:
        print("⚠ Using in-memory data storage (database not available)")
    
    # Start camera threads
    print("\nStarting camera capture threads...")
    threading.Thread(target=camera1_thread, daemon=True).start()
    threading.Thread(target=camera2_thread, daemon=True).start()
    
    # Wait a moment for cameras to initialize
    time.sleep(2)
    
    print("\nStarting LEGO Sorting System Web GUI...")
    print("Open your browser and navigate to: http://localhost:5001")
    print("=" * 50)
    
    try:
        app.run(host='0.0.0.0', port=5001, threaded=True, debug=False)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        camera1_active = False
        camera2_active = False
        if camera1:
            camera1.release()
        if camera2:
            camera2.release()
        if ROS2_AVAILABLE and rclpy.ok():
            try:
                rclpy.shutdown()
            except:
                pass
        if mqtt_client:
            try:
                mqtt_client.disconnect()
                print("✓ MQTT client disconnected")
            except:
                pass
        if db_conn:
            try:
                db_conn.close()
                print("✓ Database connection closed")
            except:
                pass
        print("Camera streams stopped.")

