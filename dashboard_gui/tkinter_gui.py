#!/usr/bin/env python3
"""
Tkinter GUI for displaying two ROS2 image topics stacked vertically
with MQTT table viewer below.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from PIL import Image as PILImage, ImageTk
import tkinter as tk
from tkinter import ttk
from threading import Thread
import queue
import sqlite3
import json
import os
import signal
import sys
from datetime import datetime

# MQTT imports
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    print("⚠ MQTT not available - install paho-mqtt: pip install paho-mqtt")

# MQTT configuration
MQTT_CONFIG = {
    'broker': os.getenv('MQTT_BROKER', 'broker.hivemq.com'),
    'port': int(os.getenv('MQTT_PORT', '1883')),
    'topic': os.getenv('MQTT_TOPIC', 'lego_sorting/sql_update'),
    'query_topic': os.getenv('MQTT_QUERY_TOPIC', 'lego_sorting/sql_query'),
    'response_topic': os.getenv('MQTT_RESPONSE_TOPIC', 'lego_sorting/sql_response'),
    'client_id': os.getenv('MQTT_CLIENT_ID', 'lego_dashboard_viewer'),
    'username': os.getenv('MQTT_USERNAME', None),
    'password': os.getenv('MQTT_PASSWORD', None)
}

# Local SQLite database file
DB_FILE = 'lego_sorting_local.db'

# Topic definitions at the top
TOPIC_1 = "/camera/image_rgb_exo"
TOPIC_2 = "/camera_annotated"

# Image display size (width, height)
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

# Table display height
TABLE_HEIGHT = 300


class ImageSubscriber(Node):
    """ROS2 node that subscribes to image topics and updates a queue."""
    
    def __init__(self, topic_name, image_queue):
        super().__init__(f'image_subscriber_{topic_name.replace("/", "_")}')
        self.topic_name = topic_name
        self.image_queue = image_queue
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            topic_name,
            self.image_callback,
            10
        )
        self.get_logger().info(f'Subscribed to {topic_name}')
    
    def image_callback(self, msg):
        """Callback function for image messages."""
        try:
            # Convert ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Keep original image (no resize to preserve aspect ratio)
            # Put image in queue (non-blocking, drop old frames)
            try:
                self.image_queue.put_nowait(cv_image)
            except queue.Full:
                # Remove old frame and add new one
                try:
                    self.image_queue.get_nowait()
                    self.image_queue.put_nowait(cv_image)
                except queue.Empty:
                    pass
        except Exception as e:
            self.get_logger().error(f'Error processing image from {self.topic_name}: {str(e)}')


def normalize_color(color):
    """Normalize color string to consistent format: first letter capitalized, rest lowercase"""
    if not color:
        return color
    return color.capitalize()


class LocalDatabase:
    """Local SQLite database manager"""
    
    def __init__(self, db_file):
        self.db_file = db_file
        self.conn = None
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database and create tables"""
        try:
            self.conn = sqlite3.connect(self.db_file, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            cursor = self.conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sorting_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    color TEXT NOT NULL,
                    aruco_marker_id INTEGER NOT NULL,
                    count INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON sorting_records(timestamp DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_status 
                ON sorting_records(status)
            """)
            
            self.conn.commit()
            print(f"✓ Local database initialized: {self.db_file}")
            
        except sqlite3.Error as e:
            print(f"⚠ Database error: {e}")
            self.conn = None
    
    def insert_record(self, color, aruco_id, count, status='Processing', timestamp=None):
        """Insert a new sorting record (color should already be normalized by caller)"""
        if not self.conn:
            return None
        
        if status not in ['Processing', 'Completed']:
            status = 'Processing'
        
        # For single robot workflow - only ONE Processing record should exist at a time
        if status == 'Processing':
            try:
                cursor = self.conn.cursor()
                cursor.execute("""
                    SELECT id, aruco_marker_id, color FROM sorting_records
                    WHERE status = 'Processing'
                    ORDER BY timestamp DESC, id DESC
                    LIMIT 1
                """)
                existing = cursor.fetchone()
                if existing:
                    print(f"⚠ Another Processing record exists (ID: {existing[0]})")
                    return None
            except Exception as e:
                print(f"⚠ Error checking for existing Processing record: {e}")
        
        try:
            cursor = self.conn.cursor()
            
            if timestamp:
                timestamp_str = timestamp if isinstance(timestamp, str) else timestamp.strftime('%Y-%m-%d %H:%M:%S')
            else:
                timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            cursor.execute("""
                INSERT INTO sorting_records (timestamp, color, aruco_marker_id, count, status)
                VALUES (?, ?, ?, ?, ?)
            """, (timestamp_str, color, aruco_id, count, status))
            
            record_id = cursor.lastrowid
            self.conn.commit()
            
            cursor.execute("""
                SELECT id, timestamp, color, aruco_marker_id, count, status
                FROM sorting_records
                WHERE id = ?
            """, (record_id,))
            
            record = cursor.fetchone()
            cursor.close()
            
            result = {
                'id': record[0],
                'timestamp': record[1],
                'color': record[2],
                'aruco_marker_id': record[3],
                'count': record[4],
                'status': record[5]
            }
            
            print(f"✓ Inserted record: ID {result['id']}, ArUco {aruco_id}, Color {color}, Status {status}, Count {count}")
            return result
            
        except sqlite3.Error as e:
            print(f"⚠ Error inserting record: {e}")
            return None
    
    def update_processing_to_completed(self, aruco_marker_id, color=None):
        """Update the most recent Processing record to Completed (color should already be normalized by caller)"""
        if not self.conn:
            return False
        
        try:
            cursor = self.conn.cursor()
            
            if color:
                cursor.execute("""
                    SELECT id FROM sorting_records
                    WHERE aruco_marker_id = ? 
                    AND color = ?
                    AND status = 'Processing'
                    ORDER BY timestamp DESC, id DESC
                    LIMIT 1
                """, (aruco_marker_id, color))
            else:
                cursor.execute("""
                    SELECT id FROM sorting_records
                    WHERE aruco_marker_id = ?
                    AND status = 'Processing'
                    ORDER BY timestamp DESC, id DESC
                    LIMIT 1
                """, (aruco_marker_id,))
            
            record_id = cursor.fetchone()
            
            if not record_id:
                cursor.close()
                return False
            
            record_id = record_id[0]
            
            cursor.execute("""
                UPDATE sorting_records
                SET status = 'Completed'
                WHERE id = ?
                AND status = 'Processing'
            """, (record_id,))
            
            self.conn.commit()
            updated = cursor.rowcount > 0
            cursor.close()
            
            if updated:
                print(f"✓ Updated Processing to Completed: ArUco ID {aruco_marker_id}, Color {color or 'any'}")
            else:
                print(f"⚠ No Processing record found to update: ArUco ID {aruco_marker_id}, Color {color or 'any'}")
            
            return updated
            
        except sqlite3.Error as e:
            print(f"⚠ Error updating Processing to Completed: {e}")
            return False
    
    def update_record_status(self, record_id, new_status):
        """Update the status of a record by ID"""
        if not self.conn:
            return False
        
        if new_status not in ['Processing', 'Completed']:
            return False
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE sorting_records
                SET status = ?
                WHERE id = ?
            """, (new_status, record_id))
            
            self.conn.commit()
            updated = cursor.rowcount > 0
            cursor.close()
            
            return updated
            
        except sqlite3.Error as e:
            print(f"⚠ Error updating record status: {e}")
            return False
    
    def get_all_records(self):
        """Get all records from database"""
        if not self.conn:
            return []
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT id, timestamp, color, aruco_marker_id, count, status
                FROM sorting_records
                ORDER BY timestamp DESC, id DESC
            """)
            
            records = cursor.fetchall()
            cursor.close()
            
            result = []
            for record in records:
                result.append({
                    'id': record[0],
                    'timestamp': record[1],
                    'color': record[2],
                    'aruco_marker_id': record[3],
                    'count': record[4],
                    'status': record[5]
                })
            
            return result
            
        except sqlite3.Error as e:
            print(f"⚠ Error fetching records: {e}")
            return []
    
    def get_latest_records(self, limit=3):
        """Get latest N records from database"""
        if not self.conn:
            return []
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT id, timestamp, color, aruco_marker_id, count, status
                FROM sorting_records
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
            """, (limit,))
            
            records = cursor.fetchall()
            cursor.close()
            
            result = []
            for record in records:
                result.append({
                    'id': record[0],
                    'timestamp': record[1],
                    'color': record[2],
                    'aruco_marker_id': record[3],
                    'count': record[4],
                    'status': record[5]
                })
            
            return result
            
        except sqlite3.Error as e:
            print(f"⚠ Error fetching records: {e}")
            return []
    
    def query_records(self, color=None, aruco_id=None):
        """Query records from database with optional filters (color should already be normalized by caller)"""
        if not self.conn:
            return []
        
        try:
            cursor = self.conn.cursor()
            
            if color and aruco_id:
                # Query by both color and aruco_id (exact match since colors are normalized)
                cursor.execute("""
                    SELECT id, timestamp, color, aruco_marker_id, count, status
                    FROM sorting_records
                    WHERE color = ? AND aruco_marker_id = ?
                    ORDER BY timestamp DESC, id DESC
                """, (color, aruco_id))
            elif color:
                # Query by color only (exact match since colors are normalized)
                cursor.execute("""
                    SELECT id, timestamp, color, aruco_marker_id, count, status
                    FROM sorting_records
                    WHERE color = ?
                    ORDER BY timestamp DESC, id DESC
                """, (color,))
            elif aruco_id:
                # Query by aruco_id only
                cursor.execute("""
                    SELECT id, timestamp, color, aruco_marker_id, count, status
                    FROM sorting_records
                    WHERE aruco_marker_id = ?
                    ORDER BY timestamp DESC, id DESC
                """, (aruco_id,))
            else:
                # Query all records
                cursor.execute("""
                    SELECT id, timestamp, color, aruco_marker_id, count, status
                    FROM sorting_records
                    ORDER BY timestamp DESC, id DESC
                """)
            
            records = cursor.fetchall()
            cursor.close()
            
            result = []
            for record in records:
                result.append({
                    'id': record[0],
                    'timestamp': record[1],
                    'color': record[2],
                    'aruco_marker_id': record[3],
                    'count': record[4],
                    'status': record[5]
                })
            
            return result
            
        except sqlite3.Error as e:
            print(f"⚠ Error querying records: {e}")
            return []
    
    def clear_database(self):
        """Clear all records from the database and reset ID sequence"""
        if not self.conn:
            return False
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM sorting_records")
            deleted_count = cursor.rowcount
            
            # Reset the AUTOINCREMENT sequence so IDs start from 1 again
            cursor.execute("DELETE FROM sqlite_sequence WHERE name='sorting_records'")
            
            self.conn.commit()
            cursor.close()
            
            print(f"✓ Cleared database: Removed {deleted_count} records, reset ID sequence")
            return True
            
        except sqlite3.Error as e:
            print(f"⚠ Error clearing database: {e}")
            return False
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


def process_mqtt_message(message_data, db):
    """Process MQTT message and update database"""
    try:
        if isinstance(message_data, str):
            data = json.loads(message_data)
        else:
            data = message_data
        
        action = data.get('action', 'insert')
        
        if action == 'update':
            aruco_id = int(data.get('aruco_marker_id'))
            color = data.get('color')
            
            if not aruco_id:
                print(f'⚠ Missing aruco_marker_id for update action')
                return False
            
            # Normalize color if provided
            if color:
                color = normalize_color(color)
            
            print(f"🔄 Processing UPDATE action: ArUco ID {aruco_id}, Color {color or 'any'}")
            return db.update_processing_to_completed(aruco_id, color)
        
        elif action == 'update_by_id':
            record_id = int(data.get('record_id'))
            new_status = str(data.get('status', 'Completed'))
            
            if not record_id:
                print(f'⚠ Missing record_id for update_by_id action')
                return False
            
            print(f"🔄 Processing UPDATE_BY_ID action: Record ID {record_id}, Status {new_status}")
            return db.update_record_status(record_id, new_status)
        
        else:
            required_fields = ['aruco_marker_id', 'color', 'status', 'count']
            if not all(field in data for field in required_fields):
                print(f'⚠ Missing required fields in message: {required_fields}')
                return False
            
            aruco_id = int(data['aruco_marker_id'])
            color = normalize_color(str(data['color']))
            status = str(data['status'])
            count = int(data['count'])
            timestamp = None
            
            if 'timestamp' in data:
                try:
                    timestamp = datetime.strptime(data['timestamp'], '%Y-%m-%d %H:%M:%S')
                except:
                    try:
                        timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
                    except:
                        pass
            
            if status == 'Completed':
                print(f"🔄 Processing INSERT with Completed status: ArUco ID {aruco_id}, Color {color}")
                updated = db.update_processing_to_completed(aruco_id, color)
                return updated
            
            print(f"🔄 Processing INSERT action: ArUco ID {aruco_id}, Color {color}, Status {status}, Count {count}")
            result = db.insert_record(
                color=color,
                aruco_id=aruco_id,
                count=count,
                status=status,
                timestamp=timestamp
            )
            
            if result is None:
                print(f"⚠ Insert failed (may be skipped due to existing Processing record)")
            
            return result is not None
                
    except json.JSONDecodeError as e:
        print(f'⚠ Invalid JSON in message: {e}')
        return False
    except Exception as e:
        print(f'⚠ Error processing message: {e}')
        import traceback
        traceback.print_exc()
        return False


def process_query_message(message_data, db, mqtt_client):
    """Process MQTT query message and publish response"""
    try:
        if isinstance(message_data, str):
            data = json.loads(message_data)
        else:
            data = message_data
        
        action = data.get('action', 'query')
        
        if action != 'query':
            print(f'⚠ Invalid action for query message: {action}')
            return
        
        # Extract optional filters
        color = data.get('color')
        aruco_id = data.get('aruco_marker_id')
        
        # Normalize color if provided
        if color:
            color = normalize_color(color)
        
        if aruco_id is not None:
            try:
                aruco_id = int(aruco_id)
            except (ValueError, TypeError):
                print(f'⚠ Invalid aruco_marker_id: {aruco_id}')
                aruco_id = None
        
        # Query database
        print(f"🔍 Processing QUERY action: Color={color}, ArUco ID={aruco_id}")
        records = db.query_records(color=color, aruco_id=aruco_id)
        
        # Prepare response
        response = {
            'status': 'success',
            'count': len(records),
            'records': records
        }
        
        # Publish response
        response_payload = json.dumps(response)
        result = mqtt_client.publish(
            MQTT_CONFIG['response_topic'],
            payload=response_payload,
            qos=1,
            retain=False
        )
        
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            print(f"✓ Query response published: {len(records)} record(s) found")
        else:
            print(f"⚠ Failed to publish query response (rc={result.rc})")
                
    except json.JSONDecodeError as e:
        print(f'⚠ Invalid JSON in query message: {e}')
        error_response = {
            'status': 'error',
            'error': f'Invalid JSON: {str(e)}',
            'count': 0,
            'records': []
        }
        try:
            mqtt_client.publish(
                MQTT_CONFIG['response_topic'],
                payload=json.dumps(error_response),
                qos=1,
                retain=False
            )
        except:
            pass
    except Exception as e:
        print(f'⚠ Error processing query message: {e}')
        import traceback
        traceback.print_exc()
        error_response = {
            'status': 'error',
            'error': str(e),
            'count': 0,
            'records': []
        }
        try:
            mqtt_client.publish(
                MQTT_CONFIG['response_topic'],
                payload=json.dumps(error_response),
                qos=1,
                retain=False
            )
        except:
            pass


class ImageGUI:
    """Tkinter GUI for displaying ROS2 image topics."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Dashboard")
        self.root.geometry(f"{DISPLAY_WIDTH + 50}x{DISPLAY_HEIGHT * 2 + TABLE_HEIGHT + 100}")
        
        # Queues for image data
        self.queue1 = queue.Queue(maxsize=2)
        self.queue2 = queue.Queue(maxsize=2)
        
        # Message queue for MQTT updates
        self.mqtt_queue = queue.Queue()
        
        # Initialize local database
        self.db = LocalDatabase(DB_FILE)
        
        # MQTT client
        self.mqtt_client = None
        self.mqtt_connected = False
        
        # Flag to track if we're shutting down (initialize early)
        self.shutting_down = False
        
        # Use PanedWindow for resizable sections
        main_paned = tk.PanedWindow(root, orient=tk.VERTICAL, sashrelief=tk.RAISED, sashwidth=5)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create PanedWindow for videos to ensure equal sizing
        video_paned = tk.PanedWindow(main_paned, orient=tk.VERTICAL, sashrelief=tk.RAISED, sashwidth=3)
        main_paned.add(video_paned, minsize=200)  # Minimum size for videos
        
        # Create labels for displaying images with equal sizing
        video_frame1 = tk.Frame(video_paned, height=DISPLAY_HEIGHT)
        video_paned.add(video_frame1, minsize=100)  # Equal minimum size
        
        video_frame2 = tk.Frame(video_paned, height=DISPLAY_HEIGHT)
        video_paned.add(video_frame2, minsize=100)  # Equal minimum size
        
        self.label1 = tk.Label(video_frame1, text=f"Waiting for {TOPIC_1}...", bg='black', fg='white')
        self.label1.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.label2 = tk.Label(video_frame2, text=f"Waiting for {TOPIC_2}...", bg='black', fg='white')
        self.label2.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create table frame (bottom pane)
        table_frame = tk.Frame(main_paned, height=TABLE_HEIGHT)
        main_paned.add(table_frame, minsize=150)  # Minimum size for table
        
        # Table title and clear button
        title_frame = tk.Frame(table_frame)
        title_frame.pack(fill=tk.X, pady=(0, 5))
        
        table_title = tk.Label(title_frame, text="LEGO Sorting Records", font=('Arial', 12, 'bold'))
        table_title.pack(side=tk.LEFT)
        
        # Clear database button
        clear_button = tk.Button(
            title_frame, 
            text="Clear Database", 
            command=self.clear_database_confirm,
            bg='#dc3545',
            fg='white',
            padx=10,
            pady=2
        )
        clear_button.pack(side=tk.RIGHT)
        
        # Create table with scrollbars
        table_container = tk.Frame(table_frame)
        table_container.pack(fill=tk.BOTH, expand=True)
        
        columns = ('ID', 'Timestamp', 'Color', 'ArUco ID', 'Count', 'Status')
        self.tree = ttk.Treeview(table_container, columns=columns, show='headings', height=4)  # Height for 3 records + header
        
        # Define column headings and widths
        self.tree.heading('ID', text='ID')
        self.tree.heading('Timestamp', text='Timestamp')
        self.tree.heading('Color', text='Color')
        self.tree.heading('ArUco ID', text='ArUco ID')
        self.tree.heading('Count', text='Count')
        self.tree.heading('Status', text='Status')
        
        self.tree.column('ID', width=50, anchor=tk.CENTER)
        self.tree.column('Timestamp', width=180, anchor=tk.W)
        self.tree.column('Color', width=100, anchor=tk.CENTER)
        self.tree.column('ArUco ID', width=80, anchor=tk.CENTER)
        self.tree.column('Count', width=80, anchor=tk.CENTER)
        self.tree.column('Status', width=120, anchor=tk.CENTER)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(table_container, orient=tk.VERTICAL, command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(table_container, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack scrollbars and tree
        self.tree.grid(row=0, column=0, sticky='nsew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')
        
        table_container.grid_rowconfigure(0, weight=1)
        table_container.grid_columnconfigure(0, weight=1)
        
        # Color tags for status
        self.tree.tag_configure('Processing', background='#fff3cd')
        self.tree.tag_configure('Completed', background='#d4edda')
        
        # Status label
        self.status_label = tk.Label(table_frame, text="Connecting to MQTT...", anchor=tk.W)
        self.status_label.pack(fill=tk.X, pady=(5, 0))
        
        # Initialize ROS2
        rclpy.init()
        
        # Create subscribers
        self.node1 = ImageSubscriber(TOPIC_1, self.queue1)
        self.node2 = ImageSubscriber(TOPIC_2, self.queue2)
        
        # Start ROS2 spinner in a separate thread
        self.ros_thread = Thread(target=self.spin_ros, daemon=True)
        self.ros_thread.start()
        
        # Initialize MQTT if available
        if MQTT_AVAILABLE:
            self.init_mqtt()
        else:
            self.status_label.config(text="⚠ MQTT not available", fg="red")
        
        # Start GUI update loops
        self.update_images()
        self.refresh_table()
        self.process_mqtt_queue()
    
    def spin_ros(self):
        """Spin ROS2 nodes in a separate thread."""
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(self.node1)
        executor.add_node(self.node2)
        try:
            while not self.shutting_down and rclpy.ok():
                executor.spin_once(timeout_sec=0.1)
        except KeyboardInterrupt:
            print("ROS2 executor interrupted")
        except Exception as e:
            print(f"⚠ ROS2 executor error: {e}")
    
    def resize_with_aspect_ratio(self, image, max_width, max_height):
        """Resize image maintaining aspect ratio."""
        original_width, original_height = image.size
        ratio = min(max_width / original_width, max_height / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        return image.resize((new_width, new_height), PILImage.Resampling.LANCZOS)
    
    def update_images(self):
        """Update displayed images from queues."""
        # Get label dimensions for proper sizing
        label1_width = self.label1.winfo_width()
        label1_height = self.label1.winfo_height()
        label2_width = self.label2.winfo_width()
        label2_height = self.label2.winfo_height()
        
        # Update image 1
        try:
            cv_image1 = self.queue1.get_nowait()
            # Convert OpenCV image to PIL Image
            rgb_image1 = cv2.cvtColor(cv_image1, cv2.COLOR_BGR2RGB)
            pil_image1 = PILImage.fromarray(rgb_image1)
            
            # Resize maintaining aspect ratio if label has valid dimensions
            if label1_width > 1 and label1_height > 1:
                pil_image1 = self.resize_with_aspect_ratio(pil_image1, label1_width, label1_height)
            else:
                # Fallback to max display size if label not yet sized
                pil_image1 = self.resize_with_aspect_ratio(pil_image1, DISPLAY_WIDTH, DISPLAY_HEIGHT)
            
            photo1 = ImageTk.PhotoImage(image=pil_image1)
            self.label1.config(image=photo1, text="")
            self.label1.image = photo1  # Keep a reference
        except queue.Empty:
            pass
        
        # Update image 2
        try:
            cv_image2 = self.queue2.get_nowait()
            # Convert OpenCV image to PIL Image
            rgb_image2 = cv2.cvtColor(cv_image2, cv2.COLOR_BGR2RGB)
            pil_image2 = PILImage.fromarray(rgb_image2)
            
            # Resize maintaining aspect ratio if label has valid dimensions
            if label2_width > 1 and label2_height > 1:
                pil_image2 = self.resize_with_aspect_ratio(pil_image2, label2_width, label2_height)
            else:
                # Fallback to max display size if label not yet sized
                pil_image2 = self.resize_with_aspect_ratio(pil_image2, DISPLAY_WIDTH, DISPLAY_HEIGHT)
            
            photo2 = ImageTk.PhotoImage(image=pil_image2)
            self.label2.config(image=photo2, text="")
            self.label2.image = photo2  # Keep a reference
        except queue.Empty:
            pass
        
        # Schedule next update
        self.root.after(33, self.update_images)  # ~30 FPS
    
    def init_mqtt(self):
        """Initialize MQTT client"""
        try:
            try:
                self.mqtt_client = mqtt.Client(
                    client_id=MQTT_CONFIG['client_id'],
                    clean_session=True,
                    callback_api_version=mqtt.CallbackAPIVersion.VERSION2
                )
            except AttributeError:
                self.mqtt_client = mqtt.Client(
                    client_id=MQTT_CONFIG['client_id'],
                    clean_session=True
                )
                def on_connect_v1(client, userdata, flags, rc):
                    self.on_mqtt_connect(client, userdata, flags, rc, None)
                def on_disconnect_v1(client, userdata, rc):
                    self.on_mqtt_disconnect(client, userdata, rc, None)
                def on_subscribe_v1(client, userdata, mid, granted_qos):
                    self.on_mqtt_subscribe(client, userdata, mid, granted_qos, None)
                self.mqtt_client.on_connect = on_connect_v1
                self.mqtt_client.on_disconnect = on_disconnect_v1
                self.mqtt_client.on_subscribe = on_subscribe_v1
            else:
                self.mqtt_client.on_connect = self.on_mqtt_connect
                self.mqtt_client.on_disconnect = self.on_mqtt_disconnect
                self.mqtt_client.on_subscribe = self.on_mqtt_subscribe
            
            self.mqtt_client.on_message = self.on_mqtt_message
            
            if MQTT_CONFIG['username'] and MQTT_CONFIG['password']:
                self.mqtt_client.username_pw_set(MQTT_CONFIG['username'], MQTT_CONFIG['password'])
            
            self.mqtt_client.connect(
                MQTT_CONFIG['broker'],
                MQTT_CONFIG['port'],
                keepalive=60
            )
            
            def mqtt_loop():
                try:
                    self.mqtt_client.loop_forever()
                except Exception as e:
                    print(f"⚠ MQTT loop error: {e}")
            
            Thread(target=mqtt_loop, daemon=True).start()
            
        except Exception as e:
            print(f"⚠ Error initializing MQTT client: {e}")
            self.status_label.config(text=f"⚠ MQTT connection failed: {e}", fg="red")
    
    def on_mqtt_connect(self, client, userdata, flags, rc, properties=None):
        """Callback when MQTT client connects"""
        if rc == 0:
            self.mqtt_connected = True
            # Subscribe to update topic
            result1, mid1 = client.subscribe(MQTT_CONFIG['topic'], qos=1)
            # Subscribe to query topic
            result2, mid2 = client.subscribe(MQTT_CONFIG['query_topic'], qos=1)
            
            if result1 == mqtt.MQTT_ERR_SUCCESS and result2 == mqtt.MQTT_ERR_SUCCESS:
                print(f"✓ MQTT connected and subscribed to:")
                print(f"  - {MQTT_CONFIG['topic']} (updates)")
                print(f"  - {MQTT_CONFIG['query_topic']} (queries)")
                self.root.after(0, lambda: self.status_label.config(
                    text="✓ Connected to MQTT", fg="green"
                ))
            else:
                self.root.after(0, lambda: self.status_label.config(
                    text="⚠ Subscription failed", fg="red"
                ))
        else:
            self.mqtt_connected = False
            self.root.after(0, lambda: self.status_label.config(
                text=f"⚠ Connection failed (code {rc})", fg="red"
            ))
    
    def on_mqtt_disconnect(self, client, userdata, rc, properties=None, *args, **kwargs):
        """Callback when MQTT client disconnects"""
        # Handle both VERSION1 and VERSION2 callbacks, plus any extra args
        self.mqtt_connected = False
        if rc != 0:
            try:
                self.root.after(0, lambda: self.status_label.config(
                    text="⚠ Disconnected", fg="red"
                ))
            except:
                pass  # Window might be destroyed
    
    def on_mqtt_subscribe(self, client, userdata, mid, granted_qos, properties=None):
        """Callback when MQTT subscription is confirmed"""
        print(f"✓ MQTT subscription confirmed (mid={mid}, QoS: {granted_qos})")
    
    def on_mqtt_message(self, client, userdata, msg):
        """Callback when MQTT message is received"""
        try:
            message_str = msg.payload.decode('utf-8')
            topic = msg.topic
            
            # Determine if this is a query or update message
            if topic == MQTT_CONFIG['query_topic']:
                # Handle query request
                print(f"📨 MQTT query received: {message_str[:100]}{'...' if len(message_str) > 100 else ''}")
                process_query_message(message_str, self.db, client)
            elif topic == MQTT_CONFIG['topic']:
                # Handle update/insert message
                print(f"📨 MQTT update received: {message_str[:100]}{'...' if len(message_str) > 100 else ''}")
                process_mqtt_message(message_str, self.db)
                # Queue GUI update
                self.mqtt_queue.put('refresh')
            else:
                print(f"⚠ Unknown topic: {topic}")
            
        except Exception as e:
            print(f"⚠ Error processing MQTT message: {e}")
    
    def process_mqtt_queue(self):
        """Process message queue for thread-safe GUI updates"""
        try:
            while True:
                msg = self.mqtt_queue.get_nowait()
                if msg == 'refresh':
                    self.refresh_table()
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.process_mqtt_queue)
    
    def refresh_table(self):
        """Refresh the table with latest 3 database records"""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Get latest 3 records from database
        records = self.db.get_latest_records(limit=3)
        
        # Add records to table
        for record in records:
            item = self.tree.insert('', 'end', values=(
                record['id'],
                record['timestamp'],
                record['color'],
                record['aruco_marker_id'],
                record['count'],
                record['status']
            ))
            
            # Apply color tag based on status
            if record['status'] == 'Processing':
                self.tree.item(item, tags=('Processing',))
            elif record['status'] == 'Completed':
                self.tree.item(item, tags=('Completed',))
        
        # Get total count for status
        all_records = self.db.get_all_records()
        total_count = len(all_records)
        
        # Update status with record count
        if self.mqtt_connected:
            self.status_label.config(text=f"✓ Connected | Showing 3 latest of {total_count} total record(s)", fg="green")
        else:
            self.status_label.config(text=f"⚠ Disconnected | Showing 3 latest of {total_count} total record(s)", fg="orange")
        
        # Schedule next refresh
        self.root.after(1000, self.refresh_table)  # Refresh every second
    
    def clear_database_confirm(self):
        """Show confirmation dialog before clearing database"""
        import tkinter.messagebox as messagebox
        result = messagebox.askyesno(
            "Clear Database",
            "Are you sure you want to clear all records from the database?\n\nThis action cannot be undone.",
            icon='warning'
        )
        if result:
            if self.db.clear_database():
                messagebox.showinfo("Success", "Database cleared successfully!")
                self.refresh_table()  # Refresh table immediately
            else:
                messagebox.showerror("Error", "Failed to clear database.")
    
    def cleanup(self):
        """Clean up all resources"""
        if hasattr(self, 'shutting_down') and self.shutting_down:
            return  # Already shutting down
        
        self.shutting_down = True
        print("\nShutting down...")
        
        # Stop MQTT client
        if self.mqtt_client:
            try:
                # Stop the loop first
                self.mqtt_client.loop_stop()
                # Disconnect (this might trigger the disconnect callback)
                self.mqtt_client.disconnect()
                print("✓ MQTT client disconnected")
            except Exception as e:
                # Ignore errors during cleanup
                pass
        
        # Close database
        if self.db:
            try:
                self.db.close()
                print("✓ Database connection closed")
            except Exception as e:
                print(f"⚠ Error closing database: {e}")
        
        # Shutdown ROS2
        try:
            if rclpy.ok():
                rclpy.shutdown()
                print("✓ ROS2 shutdown complete")
        except Exception as e:
            print(f"⚠ Error shutting down ROS2: {e}")
        
        # Destroy root window if it exists
        try:
            if hasattr(self, 'root') and self.root.winfo_exists():
                self.root.quit()
                self.root.destroy()
        except:
            pass
    
    def on_closing(self):
        """Handle window closing."""
        self.cleanup()


def main():
    """Main function to run the GUI."""
    app = None
    root = None
    
    try:
        root = tk.Tk()
        app = ImageGUI(root)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        
        # Handle Ctrl+C (SIGINT) - set up signal handler
        def signal_handler(signum, frame):
            print("\nReceived interrupt signal (Ctrl+C)")
            if app:
                # Schedule cleanup in Tkinter's event loop
                root.after(0, app.cleanup)
                root.after(100, root.quit)
            else:
                sys.exit(0)
        
        # Register signal handler for SIGINT (Ctrl+C)
        signal.signal(signal.SIGINT, signal_handler)
        
        # Run mainloop
        root.mainloop()
        
    except KeyboardInterrupt:
        print("\nReceived KeyboardInterrupt")
        if app:
            app.cleanup()
        sys.exit(0)
    except Exception as e:
        print(f"⚠ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        if app:
            app.cleanup()
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nReceived KeyboardInterrupt")
        sys.exit(0)
    except Exception as e:
        print(f"⚠ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

