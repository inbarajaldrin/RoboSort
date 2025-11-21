#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from std_msgs.msg import Float64MultiArray, Float32
from max_camera_msgs.msg import ObjectPoseArray, ObjectPose
import tkinter as tk
from tkinter import ttk
import threading
import time
import subprocess
import sys
import os
import numpy as np
import math
import xml.etree.ElementTree as ET
from ament_index_python.packages import get_package_share_directory

# Import IK functions from share/scripts directory
package_dir = get_package_share_directory('JETANK_description')
scripts_path = os.path.join(package_dir, 'scripts')
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

try:
    from ik import compute_ik, forward_kinematics, verify_solution
except ImportError:
    compute_ik = None
    forward_kinematics = None
    verify_solution = None

class JETANKGripperControlGUI(Node):
    def __init__(self):
        super().__init__('jetank_gripper_control_gui')
        
        # Load joint limits from URDF
        self.joint_limits = self.load_joint_limits_from_urdf()
        
        # Create publisher for joint states
        self.joint_state_pub = self.create_publisher(JointState, 'joint_states', 10)
        
        # Create trajectory publisher
        self.trajectory_pub = self.create_publisher(JointTrajectory, 'arm_trajectory', 10)
        
        # Create velocity controller publisher
        self.velocity_pub = self.create_publisher(Float64MultiArray, '/forward_velocity_controller/commands', 10)
        
        # Object detection data
        self.objects_data = {}  # Store latest objects data
        self.objects_lock = threading.Lock()  # Thread safety for objects data
        
        # Create subscriber for object poses
        self.objects_sub = self.create_subscription(
            ObjectPoseArray,
            '/objects_poses',
            self.objects_callback,
            10
        )
        
        # Force monitoring for gripper
        self.gripper_force_r2 = 0.0
        self.gripper_force_l2 = 0.0
        self.force_lock = threading.Lock()
        self.force_threshold = 5.0  # Default threshold
        
        # Create subscribers for gripper force topics
        self.force_r2_sub = self.create_subscription(
            Float32,
            '/gripper_r2/contact_force',
            self.force_r2_callback,
            10
        )
        
        self.force_l2_sub = self.create_subscription(
            Float32,
            '/gripper_l2/contact_force',
            self.force_l2_callback,
            10
        )
        
        # Initialize joint state message with all JETANK revolute joints
        self.joint_state = JointState()
        self.joint_state.header.frame_id = ''
        self.joint_state.name = [
            'revolute_BEARING',                  # Arm base rotation: -1.5708 to 1.5708
            'revolute_FREE_WHEEL_LEFT',          # Left free wheel: 0.0 to 6.283185
            'revolute_FREE_WHEEL_RIGHT',         # Right free wheel: 0.0 to 6.283185
            'revolute_GRIPPER_L1',               # Left gripper L1: limits from URDF
            'revolute_GRIPPER_L2',               # Left gripper L2: limits from URDF
            'Revolute_SERVO_UPPER',              # Upper arm servo: -3.1418 to 0.785594
            'Revolute_SERVO_LOWER',              # Lower arm servo: 0.0 to 1.570796
            'Revolute_DRIVING_WHEEL_R',          # Right driving wheel: 0.0 to 6.283185
            'Revolute_DRIVING_WHEEL_L',          # Left driving wheel: 0.0 to 6.283185
            'Revolute_GRIPPER_R2',               # Right gripper R2: limits from URDF
            'Revolute_GRIPPER_R1',               # Right gripper R1: limits from URDF
            'revolute_CAMERA_HOLDER_ARM_LOWER'   # Camera tilt: -0.785398 to 0.785398 (±45 degrees)
        ]
        # Initialize all joints to default positions (0.0 for most, except wheels which can be 0.0)
        self.joint_state.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.joint_state.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.joint_state.effort = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Store joint indices for easier access
        self.gripper_joint_indices = {
            'L1': 3,  # revolute_GRIPPER_L1
            'L2': 4,  # revute_GRIPPER_L2
            'R1': 10, # Revolute_GRIPPER_R1
            'R2': 9   # Revolute_GRIPPER_R2
        }
        
        self.arm_joint_indices = {
            'BEARING': 0,    # revolute_BEARING
            'SERVO_LOWER': 6, # Revolute_SERVO_LOWER
            'SERVO_UPPER': 5  # Revolute_SERVO_UPPER
        }
        
        self.camera_joint_indices = {
            'CAMERA_TILT': 11  # revolute_CAMERA_HOLDER_ARM_LOWER
        }
        
        # Threading control
        self.gui_thread = None
        self.running = True
        
        # Trajectory control
        self.trajectory_active = False
        self.trajectory_thread = None
        
        # Create Tkinter GUI in a separate thread
        self.setup_gui_thread()
        
        # Create timer for publishing joint states
        self.timer = self.create_timer(0.1, self.publish_joint_states)
        
        self.get_logger().info('JETANK Gripper Control GUI initialized')
    
    def load_joint_limits_from_urdf(self):
        """Load joint limits from URDF file dynamically"""
        joint_limits = {}
        
        try:
            # Try to get package share directory
            try:
                package_dir = get_package_share_directory('JETANK_description')
                urdf_path = os.path.join(package_dir, 'urdf', 'JETANK.urdf')
            except:
                # Fallback: try relative path from current file
                current_dir = os.path.dirname(os.path.abspath(__file__))
                urdf_path = os.path.join(current_dir, '..', 'urdf', 'JETANK.urdf')
                urdf_path = os.path.abspath(urdf_path)
            
            if not os.path.exists(urdf_path):
                self.get_logger().warn(f'URDF file not found at {urdf_path}, using default limits')
                return self.get_default_joint_limits()
            
            # Parse URDF XML
            tree = ET.parse(urdf_path)
            root = tree.getroot()
            
            # Find all joint elements
            for joint in root.findall('joint'):
                joint_name = joint.get('name')
                if joint_name is None:
                    continue
                
                # Find limit element
                limit = joint.find('limit')
                if limit is not None:
                    lower = limit.get('lower')
                    upper = limit.get('upper')
                    if lower is not None and upper is not None:
                        try:
                            joint_limits[joint_name] = {
                                'lower': float(lower),
                                'upper': float(upper)
                            }
                        except ValueError:
                            self.get_logger().warn(f'Invalid limit values for joint {joint_name}: lower={lower}, upper={upper}')
            
            self.get_logger().info(f'Loaded joint limits for {len(joint_limits)} joints from URDF')
            return joint_limits
            
        except Exception as e:
            self.get_logger().error(f'Error loading joint limits from URDF: {e}')
            return self.get_default_joint_limits()
    
    def get_default_joint_limits(self):
        """Return default joint limits as fallback"""
        return {
            'revolute_BEARING': {'lower': -1.8, 'upper': 1.8},
            'Revolute_SERVO_LOWER': {'lower': -1.892348, 'upper': 1.574492},
            'Revolute_SERVO_UPPER': {'lower': -1.548620, 'upper': 1.0},
            'revolute_CAMERA_HOLDER_ARM_LOWER': {'lower': -1.0, 'upper': 1.0},
            'revolute_GRIPPER_L1': {'lower': -1.047198, 'upper': 0.0},
            'revolute_GRIPPER_L2': {'lower': -1.047198, 'upper': 0.0},
            'Revolute_GRIPPER_R1': {'lower': 0.0, 'upper': 1.047198},
            'Revolute_GRIPPER_R2': {'lower': -1.047198, 'upper': 0.0},
        }
    
    def get_joint_limit(self, joint_name, limit_type='lower'):
        """Get joint limit for a specific joint"""
        if joint_name in self.joint_limits:
            return self.joint_limits[joint_name].get(limit_type, 0.0)
        # Fallback to default if not found
        defaults = self.get_default_joint_limits()
        if joint_name in defaults:
            return defaults[joint_name].get(limit_type, 0.0)
        return 0.0
        
    def setup_gui_thread(self):
        """Setup GUI in a separate thread"""
        self.gui_thread = threading.Thread(target=self.create_gui, daemon=True)
        self.gui_thread.start()
        
    def create_gui(self):
        """Create the Tkinter GUI in the GUI thread"""
        self.root = tk.Tk()
        self.root.title("JETANK Gripper Controller")
        self.root.geometry("450x580")
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="JETANK Gripper Control", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create Individual Control Tab
        self.create_individual_tab()
        
        # Create Arm Control Tab
        self.create_arm_control_tab()
        
        # Create Gripper Control Tab
        self.create_gripper_control_tab()
        
        # Create Motion Control Tab
        self.create_motion_control_tab()
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready", 
                                     font=('Arial', 10, 'italic'))
        self.status_label.pack(pady=10)
        
        # Start the GUI main loop
        self.root.mainloop()
        
    def create_individual_tab(self):
        """Create the individual finger control tab"""
        individual_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(individual_frame, text="Individual Control")
        
        # Arm Control
        arm_frame = ttk.LabelFrame(individual_frame, text="Arm Control", padding="10")
        arm_frame.pack(fill=tk.X, pady=5)
        
        # Bearing (Base Rotation)
        ttk.Label(arm_frame, text="Base Bearing:").pack(anchor=tk.W)
        self.bearing_var = tk.DoubleVar(value=0.0)
        bearing_lower = self.get_joint_limit('revolute_BEARING', 'lower')
        bearing_upper = self.get_joint_limit('revolute_BEARING', 'upper')
        self.bearing_scale = ttk.Scale(arm_frame, from_=bearing_lower, to=bearing_upper,
                                      variable=self.bearing_var, orient=tk.HORIZONTAL,
                                      command=self.on_bearing_change)
        self.bearing_scale.pack(fill=tk.X, pady=2)
        self.bearing_label = ttk.Label(arm_frame, text="0.000")
        self.bearing_label.pack(anchor=tk.W)
        
        # Lower Servo
        ttk.Label(arm_frame, text="Lower Servo:").pack(anchor=tk.W)
        self.servo_lower_var = tk.DoubleVar(value=0.0)
        servo_lower_lower = self.get_joint_limit('Revolute_SERVO_LOWER', 'lower')
        servo_lower_upper = self.get_joint_limit('Revolute_SERVO_LOWER', 'upper')
        self.servo_lower_scale = ttk.Scale(arm_frame, from_=servo_lower_lower, to=servo_lower_upper,
                                          variable=self.servo_lower_var, orient=tk.HORIZONTAL,
                                          command=self.on_servo_lower_change)
        self.servo_lower_scale.pack(fill=tk.X, pady=2)
        self.servo_lower_label = ttk.Label(arm_frame, text="0.000")
        self.servo_lower_label.pack(anchor=tk.W)
        
        # Upper Servo
        ttk.Label(arm_frame, text="Upper Servo:").pack(anchor=tk.W)
        self.servo_upper_var = tk.DoubleVar(value=0.0)
        servo_upper_lower = self.get_joint_limit('Revolute_SERVO_UPPER', 'lower')
        servo_upper_upper = self.get_joint_limit('Revolute_SERVO_UPPER', 'upper')
        self.servo_upper_scale = ttk.Scale(arm_frame, from_=servo_upper_lower, to=servo_upper_upper,
                                          variable=self.servo_upper_var, orient=tk.HORIZONTAL,
                                          command=self.on_servo_upper_change)
        self.servo_upper_scale.pack(fill=tk.X, pady=2)
        self.servo_upper_label = ttk.Label(arm_frame, text="0.000")
        self.servo_upper_label.pack(anchor=tk.W)
        
        # Camera Control
        camera_frame = ttk.LabelFrame(individual_frame, text="Camera Control", padding="10")
        camera_frame.pack(fill=tk.X, pady=5)
        
        # Camera Tilt
        ttk.Label(camera_frame, text="Camera Tilt (±45°):").pack(anchor=tk.W)
        self.camera_tilt_var = tk.DoubleVar(value=0.0)
        camera_lower = self.get_joint_limit('revolute_CAMERA_HOLDER_ARM_LOWER', 'lower')
        camera_upper = self.get_joint_limit('revolute_CAMERA_HOLDER_ARM_LOWER', 'upper')
        self.camera_tilt_scale = ttk.Scale(camera_frame, from_=camera_lower, to=camera_upper,
                                           variable=self.camera_tilt_var, orient=tk.HORIZONTAL,
                                           command=self.on_camera_tilt_change)
        self.camera_tilt_scale.pack(fill=tk.X, pady=2)
        self.camera_tilt_label = ttk.Label(camera_frame, text="0.000 rad (0.0°)")
        self.camera_tilt_label.pack(anchor=tk.W)
        
        # Left Gripper Control
        left_frame = ttk.LabelFrame(individual_frame, text="Left Gripper", padding="10")
        left_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(left_frame, text="L1 Finger:").pack(anchor=tk.W)
        self.left_l1_var = tk.DoubleVar(value=0.0)
        l1_lower = self.get_joint_limit('revolute_GRIPPER_L1', 'lower')
        l1_upper = self.get_joint_limit('revolute_GRIPPER_L1', 'upper')
        self.left_l1_scale = ttk.Scale(left_frame, from_=l1_lower, to=l1_upper, 
                                      variable=self.left_l1_var, orient=tk.HORIZONTAL,
                                      command=self.on_left_l1_change)
        self.left_l1_scale.pack(fill=tk.X, pady=2)
        self.left_l1_label = ttk.Label(left_frame, text="0.000")
        self.left_l1_label.pack(anchor=tk.W)
        
        ttk.Label(left_frame, text="L2 Finger:").pack(anchor=tk.W)
        self.left_l2_var = tk.DoubleVar(value=0.0)
        l2_lower = self.get_joint_limit('revolute_GRIPPER_L2', 'lower')
        l2_upper = self.get_joint_limit('revolute_GRIPPER_L2', 'upper')
        self.left_l2_scale = ttk.Scale(left_frame, from_=l2_lower, to=l2_upper,
                                      variable=self.left_l2_var, orient=tk.HORIZONTAL,
                                      command=self.on_left_l2_change)
        self.left_l2_scale.pack(fill=tk.X, pady=2)
        self.left_l2_label = ttk.Label(left_frame, text="0.000")
        self.left_l2_label.pack(anchor=tk.W)
        
        # Right Gripper Control
        right_frame = ttk.LabelFrame(individual_frame, text="Right Gripper", padding="10")
        right_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(right_frame, text="R1 Finger:").pack(anchor=tk.W)
        self.right_r1_var = tk.DoubleVar(value=0.0)
        r1_lower = self.get_joint_limit('Revolute_GRIPPER_R1', 'lower')
        r1_upper = self.get_joint_limit('Revolute_GRIPPER_R1', 'upper')
        self.right_r1_scale = ttk.Scale(right_frame, from_=r1_lower, to=r1_upper,
                                       variable=self.right_r1_var, orient=tk.HORIZONTAL,
                                       command=self.on_right_r1_change)
        self.right_r1_scale.pack(fill=tk.X, pady=2)
        self.right_r1_label = ttk.Label(right_frame, text="0.000")
        self.right_r1_label.pack(anchor=tk.W)
        
        ttk.Label(right_frame, text="R2 Finger:").pack(anchor=tk.W)
        self.right_r2_var = tk.DoubleVar(value=0.0)
        r2_lower = self.get_joint_limit('Revolute_GRIPPER_R2', 'lower')
        r2_upper = self.get_joint_limit('Revolute_GRIPPER_R2', 'upper')
        self.right_r2_scale = ttk.Scale(right_frame, from_=r2_lower, to=r2_upper,
                                       variable=self.right_r2_var, orient=tk.HORIZONTAL,
                                       command=self.on_right_r2_change)
        self.right_r2_scale.pack(fill=tk.X, pady=2)
        self.right_r2_label = ttk.Label(right_frame, text="0.000")
        self.right_r2_label.pack(anchor=tk.W)
        
        # Control buttons for individual tab
        button_frame = ttk.Frame(individual_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Reset All", 
                  command=self.reset_gripper).pack(side=tk.LEFT, padx=5)
                  
    def create_arm_control_tab(self):
        """Create the arm control tab with x, y, z coordinates"""
        arm_control_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(arm_control_frame, text="Arm Control")
        
        # Title
        title_label = ttk.Label(arm_control_frame, text="Arm Position Control", 
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Main control frame with side-by-side layout
        main_control_frame = ttk.Frame(arm_control_frame)
        main_control_frame.pack(fill=tk.X, pady=5)
        
        # Coordinate input frame (left side)
        coord_frame = ttk.LabelFrame(main_control_frame, text="Target Position", padding="10")
        coord_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # X coordinate
        x_frame = ttk.Frame(coord_frame)
        x_frame.pack(fill=tk.X, pady=2)
        ttk.Label(x_frame, text="X:", width=5).pack(side=tk.LEFT)
        self.x_var = tk.StringVar(value="0.0")
        self.x_entry = ttk.Entry(x_frame, textvariable=self.x_var, width=15)
        self.x_entry.pack(side=tk.LEFT, padx=(5, 0))
        ttk.Label(x_frame, text="mm").pack(side=tk.LEFT, padx=(5, 0))
        
        # Y coordinate
        y_frame = ttk.Frame(coord_frame)
        y_frame.pack(fill=tk.X, pady=2)
        ttk.Label(y_frame, text="Y:", width=5).pack(side=tk.LEFT)
        self.y_var = tk.StringVar(value="0.0")
        self.y_entry = ttk.Entry(y_frame, textvariable=self.y_var, width=15)
        self.y_entry.pack(side=tk.LEFT, padx=(5, 0))
        ttk.Label(y_frame, text="mm").pack(side=tk.LEFT, padx=(5, 0))
        
        # Z coordinate
        z_frame = ttk.Frame(coord_frame)
        z_frame.pack(fill=tk.X, pady=2)
        ttk.Label(z_frame, text="Z:", width=5).pack(side=tk.LEFT)
        self.z_var = tk.StringVar(value="0.0")
        self.z_entry = ttk.Entry(z_frame, textvariable=self.z_var, width=15)
        self.z_entry.pack(side=tk.LEFT, padx=(5, 0))
        ttk.Label(z_frame, text="mm").pack(side=tk.LEFT, padx=(5, 0))
        
        # Target Object frame (right side)
        object_frame = ttk.LabelFrame(main_control_frame, text="Target Object", padding="10")
        object_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Topic name input
        topic_frame = ttk.Frame(object_frame)
        topic_frame.pack(fill=tk.X, pady=2)
        ttk.Label(topic_frame, text="Topic:", width=8).pack(side=tk.LEFT)
        self.topic_var = tk.StringVar(value="/objects_poses")
        self.topic_entry = ttk.Entry(topic_frame, textvariable=self.topic_var, width=20)
        self.topic_entry.pack(side=tk.LEFT, padx=(5, 0))
        
        # Object name input
        object_name_frame = ttk.Frame(object_frame)
        object_name_frame.pack(fill=tk.X, pady=2)
        ttk.Label(object_name_frame, text="Object:", width=8).pack(side=tk.LEFT)
        self.object_name_var = tk.StringVar(value="red object_0")
        self.object_name_entry = ttk.Entry(object_name_frame, textvariable=self.object_name_var, width=20)
        self.object_name_entry.pack(side=tk.LEFT, padx=(5, 0))
        
        # Move to object button
        object_button_frame = ttk.Frame(object_frame)
        object_button_frame.pack(fill=tk.X, pady=5)
        ttk.Button(object_button_frame, text="Move to Grab", 
                  command=self.move_to_object).pack(side=tk.LEFT, padx=5)
        ttk.Button(object_button_frame, text="Refresh Objects", 
                  command=self.refresh_objects).pack(side=tk.LEFT, padx=5)
        
        # Trajectory and Camera control in side-by-side layout
        trajectory_camera_frame = ttk.Frame(arm_control_frame)
        trajectory_camera_frame.pack(fill=tk.X, pady=5)
        
        # Left side: Trajectory control
        trajectory_left_frame = ttk.Frame(trajectory_camera_frame)
        trajectory_left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Trajectory checkbox
        trajectory_frame = ttk.Frame(trajectory_left_frame)
        trajectory_frame.pack(fill=tk.X, pady=2)
        self.use_trajectory_var = tk.BooleanVar(value=True)
        self.trajectory_checkbox = ttk.Checkbutton(trajectory_frame, text="Use Trajectory Movement", 
                                                  variable=self.use_trajectory_var,
                                                  command=self.on_trajectory_checkbox_change)
        self.trajectory_checkbox.pack(side=tk.LEFT)
        
        # Duration input (always visible)
        self.duration_frame = ttk.Frame(trajectory_left_frame)
        self.duration_frame.pack(fill=tk.X, pady=2)
        ttk.Label(self.duration_frame, text="Duration (sec):").pack(side=tk.LEFT)
        self.duration_var = tk.StringVar(value="7.0")
        self.duration_entry = ttk.Entry(self.duration_frame, textvariable=self.duration_var, width=10)
        self.duration_entry.pack(side=tk.LEFT, padx=(5, 0))
        
        # Right side: Camera control in LabelFrame
        camera_control_frame = ttk.LabelFrame(trajectory_camera_frame, text="Camera", padding="10")
        camera_control_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        camera_button_frame = ttk.Frame(camera_control_frame)
        camera_button_frame.pack(fill=tk.X)
        ttk.Button(camera_button_frame, text="Down", 
                  command=self.camera_down).pack(side=tk.LEFT, padx=2)
        ttk.Button(camera_button_frame, text="Reset", 
                  command=self.camera_reset).pack(side=tk.LEFT, padx=2)
        
        # Move button
        button_frame = ttk.Frame(arm_control_frame)
        button_frame.pack(fill=tk.X, pady=10)
        ttk.Button(button_frame, text="Move to Position", 
                  command=self.move_to_position).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset Position", 
                  command=self.reset_arm_position).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Stop Trajectory", 
                  command=self.stop_trajectory).pack(side=tk.LEFT, padx=5)
        
        # Error message text box
        error_frame = ttk.LabelFrame(arm_control_frame, text="Status & Messages", padding="10")
        error_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.error_text = tk.Text(error_frame, height=8, width=50, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(error_frame, orient=tk.VERTICAL, command=self.error_text.yview)
        self.error_text.configure(yscrollcommand=scrollbar.set)
        
        self.error_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Initialize with welcome message
        self.error_text.insert(tk.END, "Arm Control Ready\n")
        self.error_text.insert(tk.END, "Check 'Use Trajectory Movement' to enable smooth timed movement\n")
        self.error_text.insert(tk.END, "Duration field is always visible - change it anytime (default: 3.0 sec)\n")
        self.error_text.insert(tk.END, "Enter target coordinates (in mm) and click 'Move to Position'\n")
        self.error_text.insert(tk.END, "Click 'Reset Position' to return to home position\n")
        self.error_text.insert(tk.END, "Inverse Kinematics will calculate the required joint angles\n\n")
        self.error_text.insert(tk.END, "Target Object:\n")
        self.error_text.insert(tk.END, "  Enter object name and click 'Move to Grab' to move to detected object\n")
        self.error_text.insert(tk.END, "  Click 'Refresh Objects' to see available objects\n")
        self.error_text.insert(tk.END, "  Object positions are automatically converted from meters to mm\n\n")
        self.error_text.insert(tk.END, "Workspace limits (approximate):\n")
        self.error_text.insert(tk.END, "  X: -200 to 200 mm\n")
        self.error_text.insert(tk.END, "  Y: -200 to 200 mm\n")
        self.error_text.insert(tk.END, "  Z: 50 to 250 mm\n\n")
        
    def create_gripper_control_tab(self):
        """Create the gripper control tab"""
        gripper_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(gripper_frame, text="Gripper Control")
        
        # Unified Gripper Control
        gripper_control_frame = ttk.LabelFrame(gripper_frame, text="Gripper Control", padding="10")
        gripper_control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(gripper_control_frame, text="Gripper:").pack(anchor=tk.W)
        self.gripper_var = tk.DoubleVar(value=0.0)
        # Servo range: 0.0 (closed) to 1.22 (open)
        # Get max joint angle from URDF dynamically (for R1, which goes 0 to max)
        gripper_max_angle = self.get_joint_limit('Revolute_GRIPPER_R1', 'upper')
        servo_max = 1.22
        self.gripper_max_angle = gripper_max_angle  # Store for use in callback
        self.gripper_scale = ttk.Scale(gripper_control_frame, from_=0.0, to=servo_max,
                                       variable=self.gripper_var, orient=tk.HORIZONTAL,
                                       command=self.on_gripper_change)
        self.gripper_scale.pack(fill=tk.X, pady=2)
        self.gripper_label = ttk.Label(gripper_control_frame, text="0.000 (Closed)")
        self.gripper_label.pack(anchor=tk.W)
        
        # Gripper trajectory control
        gripper_trajectory_frame = ttk.Frame(gripper_frame)
        gripper_trajectory_frame.pack(fill=tk.X, pady=5)
        self.use_gripper_trajectory_var = tk.BooleanVar(value=True)
        self.gripper_trajectory_checkbox = ttk.Checkbutton(gripper_trajectory_frame, text="Use Gripper Trajectory Movement", 
                                                          variable=self.use_gripper_trajectory_var,
                                                          command=self.on_gripper_trajectory_checkbox_change)
        self.gripper_trajectory_checkbox.pack(side=tk.LEFT)
        
        # Gripper duration input (always visible)
        self.gripper_duration_frame = ttk.Frame(gripper_frame)
        self.gripper_duration_frame.pack(fill=tk.X, pady=5)
        ttk.Label(self.gripper_duration_frame, text="Gripper Duration (sec):").pack(side=tk.LEFT)
        self.gripper_duration_var = tk.StringVar(value="10.0")
        self.gripper_duration_entry = ttk.Entry(self.gripper_duration_frame, textvariable=self.gripper_duration_var, width=10)
        self.gripper_duration_entry.pack(side=tk.LEFT, padx=(5, 0))
        
        # Simple control buttons
        simple_button_frame = ttk.Frame(gripper_frame)
        simple_button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(simple_button_frame, text="Open Gripper", 
                  command=self.open_gripper).pack(side=tk.LEFT, padx=5)
        ttk.Button(simple_button_frame, text="Close Gripper", 
                  command=self.close_gripper).pack(side=tk.LEFT, padx=5)
        ttk.Button(simple_button_frame, text="Get Gripper Pose", 
                  command=self.get_gripper_pose).pack(side=tk.LEFT, padx=5)
        
        # Status text box for gripper control
        gripper_status_frame = ttk.LabelFrame(gripper_frame, text="Gripper Status", padding="10")
        gripper_status_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.gripper_status_text = tk.Text(gripper_status_frame, height=6, width=50, wrap=tk.WORD)
        gripper_scrollbar = ttk.Scrollbar(gripper_status_frame, orient=tk.VERTICAL, command=self.gripper_status_text.yview)
        self.gripper_status_text.configure(yscrollcommand=gripper_scrollbar.set)
        
        self.gripper_status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        gripper_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Initialize with welcome message
        self.gripper_status_text.insert(tk.END, "Gripper Control Ready\nUse the sliders or buttons to control the gripper\n")
        
    def create_motion_control_tab(self):
        """Create the motion control tab"""
        motion_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(motion_frame, text="Motion Control")
        
        # Title
        title_label = ttk.Label(motion_frame, text="Robot Motion Control", 
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Motion control frame
        control_frame = ttk.LabelFrame(motion_frame, text="Movement Commands", padding="10")
        control_frame.pack(fill=tk.X, pady=5)
        
        # Create button grid
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(expand=True)
        
        # Forward button
        self.forward_btn = ttk.Button(button_frame, text="Forward", 
                                     command=self.move_forward,
                                     style="Accent.TButton")
        self.forward_btn.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        # Left button
        self.left_btn = ttk.Button(button_frame, text="Left", 
                                  command=self.move_left)
        self.left_btn.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        
        # Stop button
        self.stop_btn = ttk.Button(button_frame, text="STOP", 
                                   command=self.stop_motion,
                                   style="Accent.TButton")
        self.stop_btn.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        
        # Right button
        self.right_btn = ttk.Button(button_frame, text="Right", 
                                   command=self.move_right)
        self.right_btn.grid(row=1, column=2, padx=5, pady=5, sticky="nsew")
        
        # Backward button
        self.backward_btn = ttk.Button(button_frame, text="Backward", 
                                       command=self.move_backward,
                                       style="Accent.TButton")
        self.backward_btn.grid(row=2, column=1, padx=5, pady=5, sticky="nsew")
        
        # Configure grid weights for proper button sizing
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(2, weight=1)
        button_frame.rowconfigure(0, weight=1)
        button_frame.rowconfigure(1, weight=1)
        button_frame.rowconfigure(2, weight=1)
        
        # Status text box for motion control
        motion_status_frame = ttk.LabelFrame(motion_frame, text="Motion Status", padding="10")
        motion_status_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.motion_status_text = tk.Text(motion_status_frame, height=6, width=50, wrap=tk.WORD)
        motion_scrollbar = ttk.Scrollbar(motion_status_frame, orient=tk.VERTICAL, command=self.motion_status_text.yview)
        self.motion_status_text.configure(yscrollcommand=motion_scrollbar.set)
        
        self.motion_status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        motion_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Initialize with welcome message
        self.motion_status_text.insert(tk.END, "Motion Control Ready\n")
        self.motion_status_text.insert(tk.END, "Use the buttons to control robot movement:\n")
        self.motion_status_text.insert(tk.END, "• Forward: Move forward\n")
        self.motion_status_text.insert(tk.END, "• Backward: Move backward\n")
        self.motion_status_text.insert(tk.END, "• Left: Turn left\n")
        self.motion_status_text.insert(tk.END, "• Right: Turn right\n")
        self.motion_status_text.insert(tk.END, "• STOP: Stop all movement\n")
        
    def on_closing(self):
        """Handle window closing"""
        self.running = False
        self.root.destroy()
        
    def on_bearing_change(self, value):
        """Handle bearing slider change"""
        pos = float(value)
        self.bearing_label.config(text=f"{pos:.3f}")
        self.joint_state.position[self.arm_joint_indices['BEARING']] = pos
        self.update_status(f"Base Bearing: {pos:.3f}")
        
    def on_servo_lower_change(self, value):
        """Handle lower servo slider change"""
        pos = float(value)
        self.servo_lower_label.config(text=f"{pos:.3f}")
        self.joint_state.position[self.arm_joint_indices['SERVO_LOWER']] = pos
        self.update_status(f"Lower Servo: {pos:.3f}")
        
    def on_servo_upper_change(self, value):
        """Handle upper servo slider change"""
        pos = float(value)
        self.servo_upper_label.config(text=f"{pos:.3f}")
        self.joint_state.position[self.arm_joint_indices['SERVO_UPPER']] = pos
        self.update_status(f"Upper Servo: {pos:.3f}")
    
    def on_camera_tilt_change(self, value):
        """Handle camera tilt slider change"""
        pos = float(value)
        degrees = math.degrees(pos)
        self.camera_tilt_label.config(text=f"{pos:.3f} rad ({degrees:.1f}°)")
        self.joint_state.position[self.camera_joint_indices['CAMERA_TILT']] = pos
        self.update_status(f"Camera Tilt: {pos:.3f} rad ({degrees:.1f}°)")
        
    def move_to_position(self):
        """Move arm to specified x, y, z position using inverse kinematics"""
        try:
            x = float(self.x_var.get())
            y = float(self.y_var.get())
            z = float(self.z_var.get())
            
            # Check if IK functions are available
            if compute_ik is None:
                self.error_text.insert(tk.END, "Error: IK functions not available. Check ik.py import.\n")
                self.error_text.see(tk.END)
                self.update_status("Error: IK functions unavailable")
                return
            
            # Check if trajectory is enabled
            if self.use_trajectory_var.get():
                # Use trajectory movement
                duration = float(self.duration_var.get())
                
                # Check if trajectory is already active
                if self.trajectory_active:
                    self.error_text.insert(tk.END, "Error: Trajectory already in progress. Please wait.\n")
                    self.error_text.see(tk.END)
                    self.update_status("Error: Trajectory in progress")
                    return
                
                self.error_text.insert(tk.END, f"Computing IK for trajectory: X={x}mm, Y={y}mm, Z={z}mm, Duration={duration}s\n")
                self.error_text.see(tk.END)
                self.update_status(f"Computing IK trajectory for ({x}, {y}, {z})")
                
                # Compute inverse kinematics
                joint_angles = compute_ik(x, y, z, max_tries=5, position_tolerance=2.0)
                
                if joint_angles is not None:
                    # Extract joint angles
                    theta0 = joint_angles[0]  # revolute_BEARING
                    theta1 = joint_angles[1]  # Revolute_SERVO_LOWER  
                    theta3 = joint_angles[2]  # Revolute_SERVO_UPPER
                    
                    # Get current joint positions
                    current_joints = [
                        self.joint_state.position[self.arm_joint_indices['BEARING']],
                        self.joint_state.position[self.arm_joint_indices['SERVO_LOWER']],
                        self.joint_state.position[self.arm_joint_indices['SERVO_UPPER']]
                    ]
                    
                    # Target joint positions
                    target_joints = [theta0, theta1, theta3]
                    
                    # Start trajectory execution in separate thread
                    self.trajectory_active = True
                    self.trajectory_thread = threading.Thread(
                        target=self.execute_trajectory, 
                        args=(current_joints, target_joints, duration),
                        daemon=True
                    )
                    self.trajectory_thread.start()
                    
                    self.error_text.insert(tk.END, f"Trajectory started:\n")
                    self.error_text.insert(tk.END, f"  From: [{current_joints[0]:.3f}, {current_joints[1]:.3f}, {current_joints[2]:.3f}]\n")
                    self.error_text.insert(tk.END, f"  To: [{theta0:.3f}, {theta1:.3f}, {theta3:.3f}]\n")
                    self.error_text.insert(tk.END, f"  Duration: {duration}s\n\n")
                    self.error_text.see(tk.END)
                    self.update_status(f"Executing trajectory to ({x}, {y}, {z})")
                    
                else:
                    self.error_text.insert(tk.END, f"IK Failed: No solution found for target position\n")
                    self.error_text.insert(tk.END, f"Position may be outside workspace or unreachable\n")
                    self.error_text.see(tk.END)
                    self.update_status("IK Failed: Position unreachable")
            else:
                # Use instant movement
                self.error_text.insert(tk.END, f"Computing IK for position: X={x}mm, Y={y}mm, Z={z}mm\n")
                self.error_text.see(tk.END)
                self.update_status(f"Computing IK for ({x}, {y}, {z})")
                
                # Compute inverse kinematics
                joint_angles = compute_ik(x, y, z, max_tries=5, position_tolerance=2.0)
                
                if joint_angles is not None:
                    # Extract joint angles
                    theta0 = joint_angles[0]  # revolute_BEARING
                    theta1 = joint_angles[1]  # Revolute_SERVO_LOWER  
                    theta3 = joint_angles[2]  # Revolute_SERVO_UPPER
                    
                    # Update GUI sliders and joint positions
                    self.bearing_var.set(theta0)
                    self.servo_lower_var.set(theta1)
                    self.servo_upper_var.set(theta3)
                    
                    # Update joint state message
                    self.joint_state.position[self.arm_joint_indices['BEARING']] = theta0
                    self.joint_state.position[self.arm_joint_indices['SERVO_LOWER']] = theta1
                    self.joint_state.position[self.arm_joint_indices['SERVO_UPPER']] = theta3
                    
                    # Update labels
                    self.bearing_label.config(text=f"{theta0:.3f}")
                    self.servo_lower_label.config(text=f"{theta1:.3f}")
                    self.servo_upper_label.config(text=f"{theta3:.3f}")
                    
                    # Verify the solution by computing forward kinematics
                    T, actual_pos = forward_kinematics(theta0, theta1, theta3)
                    if actual_pos is not None:
                        pos_error = np.linalg.norm(np.array(actual_pos) - np.array([x, y, z]))
                        
                        self.error_text.insert(tk.END, f"IK Solution Found:\n")
                        self.error_text.insert(tk.END, f"  Base Bearing: {theta0:.4f} rad ({math.degrees(theta0):.2f}°)\n")
                        self.error_text.insert(tk.END, f"  Lower Servo: {theta1:.4f} rad ({math.degrees(theta1):.2f}°)\n")
                        self.error_text.insert(tk.END, f"  Upper Servo: {theta3:.4f} rad ({math.degrees(theta3):.2f}°)\n")
                        self.error_text.insert(tk.END, f"Target: [{x:.1f}, {y:.1f}, {z:.1f}] mm\n")
                        self.error_text.insert(tk.END, f"Actual: [{actual_pos[0]:.1f}, {actual_pos[1]:.1f}, {actual_pos[2]:.1f}] mm\n")
                        self.error_text.insert(tk.END, f"Position Error: {pos_error:.2f} mm\n\n")
                        
                        self.update_status(f"Moved to ({x}, {y}, {z}) - Error: {pos_error:.1f}mm")
                    else:
                        self.error_text.insert(tk.END, "Warning: Could not verify solution\n")
                        self.update_status(f"Moved to ({x}, {y}, {z}) - Unverified")
                    
                    self.error_text.see(tk.END)
                    
                else:
                    self.error_text.insert(tk.END, f"IK Failed: No solution found for target position\n")
                    self.error_text.insert(tk.END, f"Position may be outside workspace or unreachable\n")
                    self.error_text.insert(tk.END, f"Workspace limits (approximate):\n")
                    self.error_text.insert(tk.END, f"  X: -200 to 200 mm\n")
                    self.error_text.insert(tk.END, f"  Y: -200 to 200 mm\n")
                    self.error_text.insert(tk.END, f"  Z: 50 to 250 mm\n\n")
                    self.error_text.see(tk.END)
                    self.update_status("IK Failed: Position unreachable")
            
        except ValueError:
            self.error_text.insert(tk.END, "Error: Please enter valid numeric values for coordinates\n")
            self.error_text.see(tk.END)
            self.update_status("Error: Invalid coordinates")
        except Exception as e:
            self.error_text.insert(tk.END, f"Error: {str(e)}\n")
            self.error_text.see(tk.END)
            self.update_status(f"Error: {str(e)}")
            
    def reset_arm_position(self):
        """Reset arm to home position"""
        # Reset coordinate inputs
        self.x_var.set("0.0")
        self.y_var.set("0.0")
        self.z_var.set("0.0")
        
        # Check if trajectory is enabled
        if self.use_trajectory_var.get():
            # Use trajectory reset
            try:
                duration = float(self.duration_var.get())
                
                # Check if trajectory is already active
                if self.trajectory_active:
                    self.error_text.insert(tk.END, "Error: Trajectory already in progress. Please wait.\n")
                    self.error_text.see(tk.END)
                    self.update_status("Error: Trajectory in progress")
                    return
                
                # Get current joint positions (arm only, not camera)
                current_joints = [
                    self.joint_state.position[self.arm_joint_indices['BEARING']],
                    self.joint_state.position[self.arm_joint_indices['SERVO_LOWER']],
                    self.joint_state.position[self.arm_joint_indices['SERVO_UPPER']]
                ]
                
                # Target joint positions (home position, arm only)
                target_joints = [0.0, 0.0, 0.0]
                
                # Start trajectory execution in separate thread
                self.trajectory_active = True
                self.trajectory_thread = threading.Thread(
                    target=self.execute_reset_trajectory, 
                    args=(current_joints, target_joints, duration),
                    daemon=True
                )
                self.trajectory_thread.start()
                
                self.error_text.insert(tk.END, f"Reset trajectory started:\n")
                self.error_text.insert(tk.END, f"  From: [{current_joints[0]:.3f}, {current_joints[1]:.3f}, {current_joints[2]:.3f}]\n")
                self.error_text.insert(tk.END, f"  To: [0.000, 0.000, 0.000] (Home)\n")
                self.error_text.insert(tk.END, f"  Duration: {duration}s\n\n")
                self.error_text.see(tk.END)
                self.update_status(f"Executing reset trajectory to home position")
                
            except ValueError:
                self.error_text.insert(tk.END, "Error: Please enter valid numeric value for duration\n")
                self.error_text.see(tk.END)
                self.update_status("Error: Invalid duration")
            except Exception as e:
                self.error_text.insert(tk.END, f"Error: {str(e)}\n")
                self.error_text.see(tk.END)
                self.update_status(f"Error: {str(e)}")
        else:
            # Use instant reset (arm only, camera not affected)
            # Reset arm joints to default positions
            self.bearing_var.set(0.0)
            self.servo_lower_var.set(0.0)
            self.servo_upper_var.set(0.0)
            
            # Update joint positions (arm only)
            self.joint_state.position[self.arm_joint_indices['BEARING']] = 0.0
            self.joint_state.position[self.arm_joint_indices['SERVO_LOWER']] = 0.0
            self.joint_state.position[self.arm_joint_indices['SERVO_UPPER']] = 0.0
            
            # Update labels
            self.bearing_label.config(text="0.000")
            self.servo_lower_label.config(text="0.000")
            self.servo_upper_label.config(text="0.000")
            
            self.error_text.insert(tk.END, "Arm reset to home position (camera unchanged)\n")
            self.error_text.see(tk.END)
            self.update_status("Arm reset to home position")
    
    def camera_down(self):
        """Move camera down to -30 degrees"""
        # Convert -30 degrees to radians
        target_angle = math.radians(-30.0)
        
        # Check if trajectory is enabled
        if self.use_trajectory_var.get():
            # Use trajectory movement with fixed 3 second duration
            try:
                duration = 3.0  # Fixed duration for camera movement
                
                # Check if trajectory is already active
                if self.trajectory_active:
                    self.error_text.insert(tk.END, "Error: Trajectory already in progress. Please wait.\n")
                    self.error_text.see(tk.END)
                    self.update_status("Error: Trajectory in progress")
                    return
                
                # Get current camera position
                current_angle = self.joint_state.position[self.camera_joint_indices['CAMERA_TILT']]
                
                # Start camera trajectory execution in separate thread
                self.trajectory_active = True
                self.trajectory_thread = threading.Thread(
                    target=self.execute_camera_trajectory, 
                    args=(current_angle, target_angle, duration),
                    daemon=True
                )
                self.trajectory_thread.start()
                
                self.error_text.insert(tk.END, f"Camera down trajectory started:\n")
                self.error_text.insert(tk.END, f"  From: {math.degrees(current_angle):.1f}°\n")
                self.error_text.insert(tk.END, f"  To: -30.0°\n")
                self.error_text.insert(tk.END, f"  Duration: {duration}s\n\n")
                self.error_text.see(tk.END)
                self.update_status(f"Executing camera down trajectory")
                
            except Exception as e:
                self.error_text.insert(tk.END, f"Error: {str(e)}\n")
                self.error_text.see(tk.END)
                self.update_status(f"Error: {str(e)}")
        else:
            # Use instant movement
            self.camera_tilt_var.set(target_angle)
            self.joint_state.position[self.camera_joint_indices['CAMERA_TILT']] = target_angle
            self.camera_tilt_label.config(text=f"{target_angle:.3f} rad (-30.0°)")
            
            self.error_text.insert(tk.END, f"Camera moved to -30.0°\n")
            self.error_text.see(tk.END)
            self.update_status("Camera moved to -30.0°")
    
    def camera_reset(self):
        """Reset camera to 0 degrees"""
        target_angle = 0.0
        
        # Check if trajectory is enabled
        if self.use_trajectory_var.get():
            # Use trajectory movement with fixed 3 second duration
            try:
                duration = 3.0  # Fixed duration for camera movement
                
                # Check if trajectory is already active
                if self.trajectory_active:
                    self.error_text.insert(tk.END, "Error: Trajectory already in progress. Please wait.\n")
                    self.error_text.see(tk.END)
                    self.update_status("Error: Trajectory in progress")
                    return
                
                # Get current camera position
                current_angle = self.joint_state.position[self.camera_joint_indices['CAMERA_TILT']]
                
                # Start camera trajectory execution in separate thread
                self.trajectory_active = True
                self.trajectory_thread = threading.Thread(
                    target=self.execute_camera_trajectory, 
                    args=(current_angle, target_angle, duration),
                    daemon=True
                )
                self.trajectory_thread.start()
                
                self.error_text.insert(tk.END, f"Camera reset trajectory started:\n")
                self.error_text.insert(tk.END, f"  From: {math.degrees(current_angle):.1f}°\n")
                self.error_text.insert(tk.END, f"  To: 0.0°\n")
                self.error_text.insert(tk.END, f"  Duration: {duration}s\n\n")
                self.error_text.see(tk.END)
                self.update_status(f"Executing camera reset trajectory")
                
            except Exception as e:
                self.error_text.insert(tk.END, f"Error: {str(e)}\n")
                self.error_text.see(tk.END)
                self.update_status(f"Error: {str(e)}")
        else:
            # Use instant movement
            self.camera_tilt_var.set(target_angle)
            self.joint_state.position[self.camera_joint_indices['CAMERA_TILT']] = target_angle
            self.camera_tilt_label.config(text=f"{target_angle:.3f} rad (0.0°)")
            
            self.error_text.insert(tk.END, f"Camera reset to 0.0°\n")
            self.error_text.see(tk.END)
            self.update_status("Camera reset to 0.0°")
        
    def on_trajectory_checkbox_change(self):
        """Handle trajectory checkbox change"""
        pass
        
    def on_gripper_trajectory_checkbox_change(self):
        """Handle gripper trajectory checkbox change"""
        pass
        
    def execute_trajectory(self, start_joints, target_joints, duration):
        """Execute smooth trajectory from start to target joint positions"""
        try:
            # Calculate number of steps (50 steps per second for smooth motion)
            steps = max(10, int(duration * 50))
            dt = duration / steps
            
            # Create trajectory points
            trajectory_points = []
            
            for i in range(steps + 1):
                # Calculate interpolation factor (0 to 1)
                t = i / steps
                
                # Smooth interpolation using cubic easing
                t_smooth = 3 * t**2 - 2 * t**3  # Smooth start and end
                
                # Interpolate joint positions
                current_positions = []
                for j in range(3):
                    pos = start_joints[j] + (target_joints[j] - start_joints[j]) * t_smooth
                    current_positions.append(pos)
                
                # Create trajectory point
                point = JointTrajectoryPoint()
                point.positions = current_positions
                point.time_from_start = Duration(sec=int(t * duration), nanosec=int(((t * duration) % 1) * 1e9))
                
                trajectory_points.append(point)
            
            # Create and publish trajectory
            traj = JointTrajectory()
            traj.joint_names = ['revolute_BEARING', 'Revolute_SERVO_LOWER', 'Revolute_SERVO_UPPER']
            traj.points = trajectory_points
            
            # Publish trajectory
            self.trajectory_pub.publish(traj)
            
            # Update GUI in real-time during trajectory execution
            start_time = time.time()
            while time.time() - start_time < duration and self.trajectory_active:
                elapsed = time.time() - start_time
                t = min(1.0, elapsed / duration)
                t_smooth = 3 * t**2 - 2 * t**3
                
                # Calculate current positions
                current_positions = []
                for j in range(3):
                    pos = start_joints[j] + (target_joints[j] - start_joints[j]) * t_smooth
                    current_positions.append(pos)
                
                # Update joint state
                self.joint_state.position[self.arm_joint_indices['BEARING']] = current_positions[0]
                self.joint_state.position[self.arm_joint_indices['SERVO_LOWER']] = current_positions[1]
                self.joint_state.position[self.arm_joint_indices['SERVO_UPPER']] = current_positions[2]
                
                # Update GUI sliders (thread-safe)
                if hasattr(self, 'root') and self.root.winfo_exists():
                    self.root.after(0, self.update_gui_during_trajectory, current_positions)
                
                time.sleep(0.02)  # 50Hz update rate
            
            # Ensure final position is set
            if self.trajectory_active:
                self.joint_state.position[self.arm_joint_indices['BEARING']] = target_joints[0]
                self.joint_state.position[self.arm_joint_indices['SERVO_LOWER']] = target_joints[1]
                self.joint_state.position[self.arm_joint_indices['SERVO_UPPER']] = target_joints[2]
                
                # Update GUI to final position
                if hasattr(self, 'root') and self.root.winfo_exists():
                    self.root.after(0, self.update_gui_during_trajectory, target_joints)
                
                # Verify final position
                T, actual_pos = forward_kinematics(target_joints[0], target_joints[1], target_joints[2])
                if actual_pos is not None:
                    pos_error = np.linalg.norm(np.array(actual_pos) - np.array([float(self.x_var.get()), float(self.y_var.get()), float(self.z_var.get())]))
                    
                    if hasattr(self, 'root') and self.root.winfo_exists():
                        self.root.after(0, self.trajectory_completed, pos_error)
            
        except Exception as e:
            if hasattr(self, 'root') and self.root.winfo_exists():
                self.root.after(0, self.trajectory_error, str(e))
        finally:
            self.trajectory_active = False
            
    def update_gui_during_trajectory(self, joint_positions):
        """Update GUI sliders during trajectory execution (called from main thread)"""
        try:
            # Update sliders
            self.bearing_var.set(joint_positions[0])
            self.servo_lower_var.set(joint_positions[1])
            self.servo_upper_var.set(joint_positions[2])
            
            # Update labels
            self.bearing_label.config(text=f"{joint_positions[0]:.3f}")
            self.servo_lower_label.config(text=f"{joint_positions[1]:.3f}")
            self.servo_upper_label.config(text=f"{joint_positions[2]:.3f}")
            
        except Exception as e:
            pass  # Ignore GUI update errors during trajectory
            
    def trajectory_completed(self, pos_error):
        """Handle trajectory completion (called from main thread)"""
        self.error_text.insert(tk.END, f"Trajectory completed successfully!\n")
        self.error_text.insert(tk.END, f"Final position error: {pos_error:.2f} mm\n\n")
        self.error_text.see(tk.END)
        self.update_status(f"Trajectory completed - Error: {pos_error:.1f}mm")
        
    def trajectory_error(self, error_msg):
        """Handle trajectory error (called from main thread)"""
        self.error_text.insert(tk.END, f"Trajectory error: {error_msg}\n")
        self.error_text.see(tk.END)
        self.update_status(f"Trajectory error: {error_msg}")
        
    def execute_reset_trajectory(self, start_joints, target_joints, duration):
        """Execute smooth trajectory from current position to home position (0,0,0) - arm only, not camera"""
        try:
            # Calculate number of steps (50 steps per second for smooth motion)
            steps = max(10, int(duration * 50))
            dt = duration / steps
            
            # Create trajectory points
            trajectory_points = []
            
            for i in range(steps + 1):
                # Calculate interpolation factor (0 to 1)
                t = i / steps
                
                # Smooth interpolation using cubic easing
                t_smooth = 3 * t**2 - 2 * t**3  # Smooth start and end
                
                # Interpolate joint positions
                current_positions = []
                for j in range(3):
                    pos = start_joints[j] + (target_joints[j] - start_joints[j]) * t_smooth
                    current_positions.append(pos)
                
                # Create trajectory point
                point = JointTrajectoryPoint()
                point.positions = current_positions
                point.time_from_start = Duration(sec=int(t * duration), nanosec=int(((t * duration) % 1) * 1e9))
                
                trajectory_points.append(point)
            
            # Create and publish trajectory
            traj = JointTrajectory()
            traj.joint_names = ['revolute_BEARING', 'Revolute_SERVO_LOWER', 'Revolute_SERVO_UPPER']
            traj.points = trajectory_points
            
            # Publish trajectory
            self.trajectory_pub.publish(traj)
            
            # Update GUI in real-time during trajectory execution
            start_time = time.time()
            while time.time() - start_time < duration and self.trajectory_active:
                elapsed = time.time() - start_time
                t = min(1.0, elapsed / duration)
                t_smooth = 3 * t**2 - 2 * t**3
                
                # Calculate current positions
                current_positions = []
                for j in range(3):
                    pos = start_joints[j] + (target_joints[j] - start_joints[j]) * t_smooth
                    current_positions.append(pos)
                
                # Update joint state (arm only, camera not affected)
                self.joint_state.position[self.arm_joint_indices['BEARING']] = current_positions[0]
                self.joint_state.position[self.arm_joint_indices['SERVO_LOWER']] = current_positions[1]
                self.joint_state.position[self.arm_joint_indices['SERVO_UPPER']] = current_positions[2]
                
                # Update GUI sliders (thread-safe)
                if hasattr(self, 'root') and self.root.winfo_exists():
                    self.root.after(0, self.update_gui_during_reset_trajectory, current_positions)
                
                time.sleep(0.02)  # 50Hz update rate
            
            # Ensure final position is set
            if self.trajectory_active:
                self.joint_state.position[self.arm_joint_indices['BEARING']] = target_joints[0]
                self.joint_state.position[self.arm_joint_indices['SERVO_LOWER']] = target_joints[1]
                self.joint_state.position[self.arm_joint_indices['SERVO_UPPER']] = target_joints[2]
                
                # Update GUI to final position
                if hasattr(self, 'root') and self.root.winfo_exists():
                    self.root.after(0, self.update_gui_during_reset_trajectory, target_joints)
                    self.root.after(0, self.reset_trajectory_completed)
            
        except Exception as e:
            if hasattr(self, 'root') and self.root.winfo_exists():
                self.root.after(0, self.reset_trajectory_error, str(e))
        finally:
            self.trajectory_active = False
            
    def update_gui_during_reset_trajectory(self, joint_positions):
        """Update GUI sliders during reset trajectory execution (called from main thread)"""
        try:
            # Update sliders (arm only, camera not affected)
            self.bearing_var.set(joint_positions[0])
            self.servo_lower_var.set(joint_positions[1])
            self.servo_upper_var.set(joint_positions[2])
            
            # Update labels
            self.bearing_label.config(text=f"{joint_positions[0]:.3f}")
            self.servo_lower_label.config(text=f"{joint_positions[1]:.3f}")
            self.servo_upper_label.config(text=f"{joint_positions[2]:.3f}")
            
        except Exception as e:
            pass  # Ignore GUI update errors during trajectory
            
    def reset_trajectory_completed(self):
        """Handle reset trajectory completion (called from main thread)"""
        self.error_text.insert(tk.END, f"Reset trajectory completed successfully!\n")
        self.error_text.insert(tk.END, f"Arm reset to home position (camera unchanged)\n\n")
        self.error_text.see(tk.END)
        self.update_status("Reset trajectory completed - Arm at home position")
        
    def reset_trajectory_error(self, error_msg):
        """Handle reset trajectory error (called from main thread)"""
        self.error_text.insert(tk.END, f"Reset trajectory error: {error_msg}\n")
        self.error_text.see(tk.END)
        self.update_status(f"Reset trajectory error: {error_msg}")
    
    def execute_camera_trajectory(self, start_angle, target_angle, duration):
        """Execute smooth trajectory for camera joint only"""
        try:
            # Calculate number of steps (50 steps per second for smooth motion)
            steps = max(10, int(duration * 50))
            
            # Update GUI in real-time during trajectory execution
            start_time = time.time()
            while time.time() - start_time < duration and self.trajectory_active:
                elapsed = time.time() - start_time
                t = min(1.0, elapsed / duration)
                
                # Smooth interpolation using cubic easing
                t_smooth = 3 * t**2 - 2 * t**3
                
                # Calculate current camera angle
                current_angle = start_angle + (target_angle - start_angle) * t_smooth
                
                # Update joint state
                self.joint_state.position[self.camera_joint_indices['CAMERA_TILT']] = current_angle
                
                # Update GUI slider (thread-safe)
                if hasattr(self, 'root') and self.root.winfo_exists():
                    self.root.after(0, self.update_gui_during_camera_trajectory, current_angle)
                
                time.sleep(0.02)  # 50Hz update rate
            
            # Ensure final position is set
            if self.trajectory_active:
                self.joint_state.position[self.camera_joint_indices['CAMERA_TILT']] = target_angle
                
                # Update GUI to final position
                if hasattr(self, 'root') and self.root.winfo_exists():
                    self.root.after(0, self.update_gui_during_camera_trajectory, target_angle)
                    self.root.after(0, self.camera_trajectory_completed)
            
        except Exception as e:
            if hasattr(self, 'root') and self.root.winfo_exists():
                self.root.after(0, self.camera_trajectory_error, str(e))
        finally:
            self.trajectory_active = False
    
    def update_gui_during_camera_trajectory(self, camera_angle):
        """Update GUI slider during camera trajectory execution (called from main thread)"""
        try:
            # Update slider
            self.camera_tilt_var.set(camera_angle)
            
            # Update label
            degrees = math.degrees(camera_angle)
            self.camera_tilt_label.config(text=f"{camera_angle:.3f} rad ({degrees:.1f}°)")
            
        except Exception as e:
            pass  # Ignore GUI update errors during trajectory
    
    def camera_trajectory_completed(self):
        """Handle camera trajectory completion (called from main thread)"""
        self.error_text.insert(tk.END, f"Camera trajectory completed successfully!\n\n")
        self.error_text.see(tk.END)
        self.update_status("Camera trajectory completed")
    
    def camera_trajectory_error(self, error_msg):
        """Handle camera trajectory error (called from main thread)"""
        self.error_text.insert(tk.END, f"Camera trajectory error: {error_msg}\n")
        self.error_text.see(tk.END)
        self.update_status(f"Camera trajectory error: {error_msg}")
        
    def execute_gripper_trajectory(self, start_joints, target_joints, duration):
        """Execute smooth trajectory for gripper joints with force monitoring"""
        try:
            # Determine if we're closing (target is more closed than start)
            # For closing: target positions are closer to 0.0 than start positions
            is_closing = False
            if len(target_joints) == 4 and len(start_joints) == 4:
                # Check if average target position is closer to 0 than start
                avg_start = sum(abs(s) for s in start_joints) / 4.0
                avg_target = sum(abs(t) for t in target_joints) / 4.0
                is_closing = avg_target < avg_start
            
            # Calculate number of steps (50 steps per second for smooth motion)
            steps = max(10, int(duration * 50))
            dt = duration / steps
            
            # Create trajectory points
            trajectory_points = []
            
            for i in range(steps + 1):
                # Calculate interpolation factor (0 to 1)
                t = i / steps
                
                # Smooth interpolation using cubic easing
                t_smooth = 3 * t**2 - 2 * t**3  # Smooth start and end
                
                # Interpolate joint positions
                current_positions = []
                for j in range(4):  # 4 gripper joints
                    pos = start_joints[j] + (target_joints[j] - start_joints[j]) * t_smooth
                    current_positions.append(pos)
                
                # Create trajectory point
                point = JointTrajectoryPoint()
                point.positions = current_positions
                point.time_from_start = Duration(sec=int(t * duration), nanosec=int(((t * duration) % 1) * 1e9))
                
                trajectory_points.append(point)
            
            # Create and publish trajectory
            traj = JointTrajectory()
            traj.joint_names = ['revolute_GRIPPER_L1', 'revolute_GRIPPER_L2', 'Revolute_GRIPPER_R1', 'Revolute_GRIPPER_R2']
            traj.points = trajectory_points
            
            # Publish trajectory
            self.trajectory_pub.publish(traj)
            
            # Update GUI in real-time during trajectory execution
            start_time = time.time()
            force_exceeded = False
            while time.time() - start_time < duration and self.trajectory_active:
                elapsed = time.time() - start_time
                t = min(1.0, elapsed / duration)
                t_smooth = 3 * t**2 - 2 * t**3
                
                # Check forces if closing
                if is_closing:
                    force_r2, force_l2 = self.get_gripper_forces()
                    max_force = max(force_r2, force_l2)
                    
                    if max_force >= self.force_threshold:
                            # Force threshold exceeded - stop closing
                            force_exceeded = True
                            self.trajectory_active = False
                            
                            # Get current positions at stopping point
                            current_positions = []
                            for j in range(4):
                                pos = start_joints[j] + (target_joints[j] - start_joints[j]) * t_smooth
                                current_positions.append(pos)
                            
                            # Update joint state to current position
                            self.joint_state.position[self.gripper_joint_indices['L1']] = current_positions[0]
                            self.joint_state.position[self.gripper_joint_indices['L2']] = current_positions[1]
                            self.joint_state.position[self.gripper_joint_indices['R1']] = current_positions[2]
                            self.joint_state.position[self.gripper_joint_indices['R2']] = current_positions[3]
                            
                            # Notify GUI
                            if hasattr(self, 'root') and self.root.winfo_exists():
                                self.root.after(0, self.gripper_force_stopped, current_positions, max_force)
                            
                            break
                
                # Calculate current positions
                current_positions = []
                for j in range(4):
                    pos = start_joints[j] + (target_joints[j] - start_joints[j]) * t_smooth
                    current_positions.append(pos)
                
                # Update joint state
                self.joint_state.position[self.gripper_joint_indices['L1']] = current_positions[0]
                self.joint_state.position[self.gripper_joint_indices['L2']] = current_positions[1]
                self.joint_state.position[self.gripper_joint_indices['R1']] = current_positions[2]
                self.joint_state.position[self.gripper_joint_indices['R2']] = current_positions[3]
                
                # Update GUI sliders (thread-safe)
                if hasattr(self, 'root') and self.root.winfo_exists():
                    self.root.after(0, self.update_gripper_gui_during_trajectory, current_positions)
                
                time.sleep(0.02)  # 50Hz update rate
            
            # Ensure final position is set
            if self.trajectory_active:
                self.joint_state.position[self.gripper_joint_indices['L1']] = target_joints[0]
                self.joint_state.position[self.gripper_joint_indices['L2']] = target_joints[1]
                self.joint_state.position[self.gripper_joint_indices['R1']] = target_joints[2]
                self.joint_state.position[self.gripper_joint_indices['R2']] = target_joints[3]
                
                # Update GUI to final position
                if hasattr(self, 'root') and self.root.winfo_exists():
                    self.root.after(0, self.update_gripper_gui_during_trajectory, target_joints)
                    self.root.after(0, self.gripper_trajectory_completed)
            
        except Exception as e:
            if hasattr(self, 'root') and self.root.winfo_exists():
                self.root.after(0, self.gripper_trajectory_error, str(e))
        finally:
            self.trajectory_active = False
            
    def update_gripper_gui_during_trajectory(self, joint_positions):
        """Update gripper GUI sliders during trajectory execution (called from main thread)"""
        try:
            # Update sliders with proper mappings
            l1_pos = joint_positions[0]
            l2_pos = joint_positions[1]
            r1_pos = joint_positions[2]
            r2_pos = joint_positions[3]
            
            # Update unified gripper slider
            # Convert joint angle back to servo value: joint_angle → servo_value
            # joint_angle = (servo_value / 1.22) * max_angle, so servo_value = (joint_angle / max_angle) * 1.22
            joint_angle = abs(r1_pos) if r1_pos >= 0 else abs(l1_pos)
            max_angle = self.get_joint_limit('Revolute_GRIPPER_R1', 'upper')
            servo_value = (joint_angle / max_angle) * 1.22 if joint_angle > 0 and max_angle > 0 else 0.0
            if hasattr(self, 'gripper_var'):
                self.gripper_var.set(servo_value)
            
            # Update joint state
            self.joint_state.position[self.gripper_joint_indices['L1']] = l1_pos
            self.joint_state.position[self.gripper_joint_indices['L2']] = l2_pos
            self.joint_state.position[self.gripper_joint_indices['R1']] = r1_pos
            self.joint_state.position[self.gripper_joint_indices['R2']] = r2_pos
            
            # Update unified gripper label (show servo value)
            if hasattr(self, 'gripper_label'):
                status = "Open" if servo_value > 0.1 else "Closed"
                self.gripper_label.config(text=f"{servo_value:.3f} ({status})")
            
        except Exception as e:
            pass  # Ignore GUI update errors during trajectory
            
    def gripper_trajectory_completed(self):
        """Handle gripper trajectory completion (called from main thread)"""
        self.gripper_status_text.insert(tk.END, f"Gripper trajectory completed successfully!\n\n")
        self.gripper_status_text.see(tk.END)
        self.update_status("Gripper trajectory completed")
    
    def gripper_force_stopped(self, current_positions, max_force):
        """Handle gripper stopping due to force threshold (called from main thread)"""
        self.gripper_status_text.insert(tk.END, f"Gripper stopped due to force threshold!\n")
        self.gripper_status_text.insert(tk.END, f"  Max force detected: {max_force:.3f} (threshold: {self.force_threshold})\n")
        self.gripper_status_text.insert(tk.END, f"  Stopped at position: L1={current_positions[0]:.3f}, L2={current_positions[1]:.3f}, R1={current_positions[2]:.3f}, R2={current_positions[3]:.3f}\n\n")
        self.gripper_status_text.see(tk.END)
        self.update_status(f"Gripper stopped - Force threshold exceeded ({max_force:.2f})")
        
    def gripper_trajectory_error(self, error_msg):
        """Handle gripper trajectory error (called from main thread)"""
        self.gripper_status_text.insert(tk.END, f"Gripper trajectory error: {error_msg}\n")
        self.gripper_status_text.see(tk.END)
        self.update_status(f"Gripper trajectory error: {error_msg}")
        
    def stop_trajectory(self):
        """Stop current trajectory execution"""
        if self.trajectory_active:
            self.trajectory_active = False
            self.error_text.insert(tk.END, "Trajectory stopped by user\n")
            self.error_text.see(tk.END)
            self.update_status("Trajectory stopped")
        else:
            self.error_text.insert(tk.END, "No trajectory is currently running\n")
            self.error_text.see(tk.END)
            self.update_status("No active trajectory")
        
    def get_gripper_pose(self):
        """Get gripper pose using tf2_echo command"""
        try:
            # Set ROS domain ID
            import os
            os.environ['ROS_DOMAIN_ID'] = '6'
            
            # Run tf2_echo command to get pose from BEARING_1 to GRIPPER_CENTER_LINK
            cmd = ["ros2", "run", "tf2_ros", "tf2_echo", "BEARING_1", "GRIPPER_CENTER_LINK"]
            
            # Start the process
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True,
                env=os.environ.copy()
            )
            
            # Read output line by line with timeout
            output_lines = []
            start_time = time.time()
            timeout = 8  # 8 second timeout
            
            while time.time() - start_time < timeout:
                # Check if process is still running
                if process.poll() is not None:
                    break
                    
                # Try to read a line
                try:
                    line = process.stdout.readline()
                    if line:
                        line = line.strip()
                        output_lines.append(line)
                        
                        # Check if we got the transform data
                        if "Translation:" in line:
                            # Read a few more lines to get complete data
                            for _ in range(3):
                                try:
                                    next_line = process.stdout.readline()
                                    if next_line:
                                        next_line = next_line.strip()
                                        output_lines.append(next_line)
                                except:
                                    break
                            break
                            
                except Exception as e:
                    break
            
            # Kill the process if it's still running
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
            
            # Parse the output
            for line in output_lines:
                if "Translation:" in line:
                    # Extract position values
                    try:
                        pos_data = line.split("Translation:")[1].strip()
                        # Remove brackets and split by comma
                        pos_data = pos_data.strip('[]')
                        pos_values = [val.strip() for val in pos_data.split(',')]
                        if len(pos_values) >= 3:
                            x = float(pos_values[0])
                            y = float(pos_values[1])
                            z = float(pos_values[2])
                            
                            # Update the Arm Control tab coordinates
                            if hasattr(self, 'x_var'):
                                self.x_var.set(f"{x*1000:.1f}")  # Convert to mm
                                self.y_var.set(f"{y*1000:.1f}")
                                self.z_var.set(f"{z*1000:.1f}")
                            
                            # Show in gripper status text box
                            if hasattr(self, 'gripper_status_text'):
                                self.gripper_status_text.insert(tk.END, f"Current gripper pose: X={x*1000:.1f}mm, Y={y*1000:.1f}mm, Z={z*1000:.1f}mm\n")
                                self.gripper_status_text.see(tk.END)
                            
                            self.update_status(f"Gripper pose: X={x*1000:.1f}mm, Y={y*1000:.1f}mm, Z={z*1000:.1f}mm")
                            return
                    except Exception as e:
                        pass
            
            # If we get here, no valid position was found
            self.update_status("Error: Could not get gripper pose")
            if hasattr(self, 'gripper_status_text'):
                self.gripper_status_text.insert(tk.END, "Error: Could not parse gripper pose from tf2_echo output\n")
                self.gripper_status_text.see(tk.END)
                    
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            if hasattr(self, 'gripper_status_text'):
                self.gripper_status_text.insert(tk.END, f"Error getting gripper pose: {str(e)}\n")
                self.gripper_status_text.see(tk.END)
        
    def on_left_l1_change(self, value):
        """Handle left L1 slider change"""
        pos = float(value)
        self.left_l1_label.config(text=f"{pos:.3f}")
        self.joint_state.position[self.gripper_joint_indices['L1']] = pos
        self.update_status(f"Left L1: {pos:.3f}")
        
    def on_left_l2_change(self, value):
        """Handle left L2 slider change"""
        pos = float(value)
        self.left_l2_label.config(text=f"{pos:.3f}")
        self.joint_state.position[self.gripper_joint_indices['L2']] = pos
        self.update_status(f"Left L2: {pos:.3f}")
        
    def on_right_r1_change(self, value):
        """Handle right R1 slider change"""
        pos = float(value)
        self.right_r1_label.config(text=f"{pos:.3f}")
        self.joint_state.position[self.gripper_joint_indices['R1']] = pos
        self.update_status(f"Right R1: {pos:.3f}")
        
    def on_right_r2_change(self, value):
        """Handle right R2 slider change"""
        pos = float(value)
        self.right_r2_label.config(text=f"{pos:.3f}")
        self.joint_state.position[self.gripper_joint_indices['R2']] = pos
        self.update_status(f"Right R2: {pos:.3f}")
        
    def on_gripper_change(self, value):
        """Handle unified gripper slider change (controls both left and right grippers)"""
        servo_value = float(value)
        # Slider: 0.0 (closed) to 1.22 (open) - servo range
        # Map servo value to joint angle: 0.0 → 0.0, 1.22 → max_angle (from URDF)
        max_angle = self.get_joint_limit('Revolute_GRIPPER_R1', 'upper')
        joint_angle = (servo_value / 1.22) * max_angle
        
        # Left gripper: map to negative (0.0 → 0.0, max_angle → -max_angle)
        # Right gripper: map to positive (0.0 → 0.0, max_angle → max_angle)
        
        # Left side: both L1 and L2 get negative value
        left_pos = -joint_angle
        self.joint_state.position[self.gripper_joint_indices['L1']] = left_pos
        self.joint_state.position[self.gripper_joint_indices['L2']] = left_pos
        
        # Right side: R1 gets positive value, R2 gets negative value
        right_r1_pos = joint_angle
        right_r2_pos = -joint_angle
        self.joint_state.position[self.gripper_joint_indices['R1']] = right_r1_pos
        self.joint_state.position[self.gripper_joint_indices['R2']] = right_r2_pos
        
        # Update individual sliders if they exist
        if hasattr(self, 'left_l1_var'):
            self.left_l1_var.set(left_pos)
            self.left_l2_var.set(left_pos)
        if hasattr(self, 'right_r1_var'):
            self.right_r1_var.set(right_r1_pos)
            self.right_r2_var.set(right_r2_pos)
        
        # Update labels - show servo value (0-1.22)
        status = "Open" if servo_value > 0.1 else "Closed"
        self.gripper_label.config(text=f"{servo_value:.3f} ({status})")
        if hasattr(self, 'left_l1_label'):
            self.left_l1_label.config(text=f"{left_pos:.3f}")
            self.left_l2_label.config(text=f"{left_pos:.3f}")
        if hasattr(self, 'right_r1_label'):
            self.right_r1_label.config(text=f"{right_r1_pos:.3f}")
            self.right_r2_label.config(text=f"{right_r2_pos:.3f}")
        
        self.update_status(f"Gripper: {servo_value:.3f} ({status})")
        
    def open_gripper(self):
        """Open gripper using simple controls"""
        # Check if gripper trajectory is enabled
        if self.use_gripper_trajectory_var.get():
            # Use trajectory movement
            try:
                duration = float(self.gripper_duration_var.get())
                
                # Check if trajectory is already active
                if self.trajectory_active:
                    self.gripper_status_text.insert(tk.END, "Error: Trajectory already in progress. Please wait.\n")
                    self.gripper_status_text.see(tk.END)
                    self.update_status("Error: Trajectory in progress")
                    return
                
                # Get current gripper joint positions
                current_gripper_joints = [
                    self.joint_state.position[self.gripper_joint_indices['L1']],
                    self.joint_state.position[self.gripper_joint_indices['L2']],
                    self.joint_state.position[self.gripper_joint_indices['R1']],
                    self.joint_state.position[self.gripper_joint_indices['R2']]
                ]
                
                # Target gripper joint positions (open) - use dynamic limits from URDF
                max_angle = self.get_joint_limit('Revolute_GRIPPER_R1', 'upper')
                target_gripper_joints = [-max_angle, -max_angle, max_angle, -max_angle]
                
                # Start gripper trajectory execution in separate thread
                self.trajectory_active = True
                self.trajectory_thread = threading.Thread(
                    target=self.execute_gripper_trajectory, 
                    args=(current_gripper_joints, target_gripper_joints, duration),
                    daemon=True
                )
                self.trajectory_thread.start()
                
                self.gripper_status_text.insert(tk.END, f"Gripper open trajectory started:\n")
                self.gripper_status_text.insert(tk.END, f"  Duration: {duration}s\n\n")
                self.gripper_status_text.see(tk.END)
                self.update_status(f"Executing gripper open trajectory")
                
            except ValueError:
                self.gripper_status_text.insert(tk.END, "Error: Please enter valid numeric value for gripper duration\n")
                self.gripper_status_text.see(tk.END)
                self.update_status("Error: Invalid gripper duration")
            except Exception as e:
                self.gripper_status_text.insert(tk.END, f"Error: {str(e)}\n")
                self.gripper_status_text.see(tk.END)
                self.update_status(f"Error: {str(e)}")
        else:
            # Use instant movement
            # Set unified gripper slider to open position (1.22 - servo max)
            if hasattr(self, 'gripper_var'):
                self.gripper_var.set(1.22)
            
            # Update joint positions with proper mappings - use dynamic limits from URDF
            max_angle = self.get_joint_limit('Revolute_GRIPPER_R1', 'upper')
            self.joint_state.position[self.gripper_joint_indices['L1']] = -max_angle  # L1
            self.joint_state.position[self.gripper_joint_indices['L2']] = -max_angle  # L2
            self.joint_state.position[self.gripper_joint_indices['R1']] = max_angle   # R1 (direct mapping)
            self.joint_state.position[self.gripper_joint_indices['R2']] = -max_angle  # R2 (inverted mapping)
            
            # Update gripper label
            if hasattr(self, 'gripper_label'):
                self.gripper_label.config(text="1.220 (Open)")
            
            # Update individual tab sliders if they exist
            if hasattr(self, 'left_l1_var'):
                self.left_l1_var.set(-max_angle)
                self.left_l2_var.set(-max_angle)
                self.right_r1_var.set(max_angle)
                self.right_r2_var.set(-max_angle)
                
            # Update labels
            if hasattr(self, 'simple_left_label'):
                self.simple_left_label.config(text=f"{-max_angle:.3f} (Open)")
            if hasattr(self, 'simple_right_label'):
                self.simple_right_label.config(text=f"{max_angle:.3f} (Open)")
            
            self.update_status("Gripper opened (simple control)")
        
    def close_gripper(self):
        """Close gripper using simple controls"""
        # Check if gripper trajectory is enabled
        if self.use_gripper_trajectory_var.get():
            # Use trajectory movement
            try:
                duration = float(self.gripper_duration_var.get())
                
                # Check if trajectory is already active
                if self.trajectory_active:
                    self.gripper_status_text.insert(tk.END, "Error: Trajectory already in progress. Please wait.\n")
                    self.gripper_status_text.see(tk.END)
                    self.update_status("Error: Trajectory in progress")
                    return
                
                # Get current gripper joint positions
                current_gripper_joints = [
                    self.joint_state.position[self.gripper_joint_indices['L1']],
                    self.joint_state.position[self.gripper_joint_indices['L2']],
                    self.joint_state.position[self.gripper_joint_indices['R1']],
                    self.joint_state.position[self.gripper_joint_indices['R2']]
                ]
                
                # Target gripper joint positions (closed)
                target_gripper_joints = [0.0, 0.0, 0.0, 0.0]
                
                # Start gripper trajectory execution in separate thread
                self.trajectory_active = True
                self.trajectory_thread = threading.Thread(
                    target=self.execute_gripper_trajectory, 
                    args=(current_gripper_joints, target_gripper_joints, duration),
                    daemon=True
                )
                self.trajectory_thread.start()
                
                self.gripper_status_text.insert(tk.END, f"Gripper close trajectory started:\n")
                self.gripper_status_text.insert(tk.END, f"  Duration: {duration}s\n\n")
                self.gripper_status_text.see(tk.END)
                self.update_status(f"Executing gripper close trajectory")
                
            except ValueError:
                self.gripper_status_text.insert(tk.END, "Error: Please enter valid numeric value for gripper duration\n")
                self.gripper_status_text.see(tk.END)
                self.update_status("Error: Invalid gripper duration")
            except Exception as e:
                self.gripper_status_text.insert(tk.END, f"Error: {str(e)}\n")
                self.gripper_status_text.see(tk.END)
                self.update_status(f"Error: {str(e)}")
        else:
            # Use instant movement
            # Set unified gripper slider to closed position (0.0)
            if hasattr(self, 'gripper_var'):
                self.gripper_var.set(0.0)
            
            # Update joint positions
            self.joint_state.position[self.gripper_joint_indices['L1']] = 0.0  # L1
            self.joint_state.position[self.gripper_joint_indices['L2']] = 0.0  # L2
            self.joint_state.position[self.gripper_joint_indices['R1']] = 0.0  # R1
            
            # Update gripper label
            if hasattr(self, 'gripper_label'):
                self.gripper_label.config(text="0.000 (Closed)")
            self.joint_state.position[self.gripper_joint_indices['R2']] = 0.0  # R2
            
            # Update individual tab sliders if they exist
            if hasattr(self, 'left_l1_var'):
                self.left_l1_var.set(0.0)
                self.left_l2_var.set(0.0)
                self.right_r1_var.set(0.0)
                self.right_r2_var.set(0.0)
                
            # Update labels
            self.simple_left_label.config(text="0.000 (Closed)")
            self.simple_right_label.config(text="0.000 (Closed)")
            
            self.update_status("Gripper closed (simple control)")
        
        
    def reset_gripper(self):
        """Reset gripper and arm to default position"""
        # Reset gripper joints to closed position
        self.left_l1_var.set(0.0)
        self.left_l2_var.set(0.0)
        self.right_r1_var.set(0.0)
        self.right_r2_var.set(0.0)
        
        # Update joint positions
        self.joint_state.position[self.gripper_joint_indices['L1']] = 0.0
        self.joint_state.position[self.gripper_joint_indices['L2']] = 0.0
        self.joint_state.position[self.gripper_joint_indices['R1']] = 0.0
        self.joint_state.position[self.gripper_joint_indices['R2']] = 0.0
        
        # Update labels
        self.left_l1_label.config(text="0.000")
        self.left_l2_label.config(text="0.000")
        self.right_r1_label.config(text="0.000")
        self.right_r2_label.config(text="0.000")
        
        # Reset arm joints
        self.bearing_var.set(0.0)
        self.servo_lower_var.set(0.0)
        self.servo_upper_var.set(0.0)
        
        # Reset camera tilt
        self.camera_tilt_var.set(0.0)
        
        # Update joint positions
        self.joint_state.position[self.arm_joint_indices['BEARING']] = 0.0
        self.joint_state.position[self.arm_joint_indices['SERVO_LOWER']] = 0.0
        self.joint_state.position[self.arm_joint_indices['SERVO_UPPER']] = 0.0
        self.joint_state.position[self.camera_joint_indices['CAMERA_TILT']] = 0.0
        
        # Update labels
        self.bearing_label.config(text="0.000")
        self.servo_lower_label.config(text="0.000")
        self.servo_upper_label.config(text="0.000")
        self.camera_tilt_label.config(text="0.000 rad (0.0°)")
        
        self.update_status("Gripper, arm, and camera reset")
        
    def move_forward(self):
        """Send forward motion command"""
        msg = Float64MultiArray()
        msg.data = [-5.0, 5.0, -5.0, 5.0]
        self.velocity_pub.publish(msg)
        self.motion_status_text.insert(tk.END, "Forward command sent: [-5.0, 5.0, -5.0, 5.0]\n")
        self.motion_status_text.see(tk.END)
        self.update_status("Moving forward")
        
    def move_backward(self):
        """Send backward motion command"""
        msg = Float64MultiArray()
        msg.data = [5.0, -5.0, 5.0, -5.0]
        self.velocity_pub.publish(msg)
        self.motion_status_text.insert(tk.END, "Backward command sent: [5.0, -5.0, 5.0, -5.0]\n")
        self.motion_status_text.see(tk.END)
        self.update_status("Moving backward")
        
    def move_left(self):
        """Send left turn command"""
        msg = Float64MultiArray()
        msg.data = [5.0, 5.0, 5.0, 5.0]
        self.velocity_pub.publish(msg)
        self.motion_status_text.insert(tk.END, "Left turn command sent: [5.0, 5.0, 5.0, 5.0]\n")
        self.motion_status_text.see(tk.END)
        self.update_status("Turning left")
        
    def move_right(self):
        """Send right turn command"""
        msg = Float64MultiArray()
        msg.data = [-5.0, -5.0, -5.0, -5.0]
        self.velocity_pub.publish(msg)
        self.motion_status_text.insert(tk.END, "Right turn command sent: [-5.0, -5.0, -5.0, -5.0]\n")
        self.motion_status_text.see(tk.END)
        self.update_status("Turning right")
        
    def stop_motion(self):
        """Send stop motion command"""
        msg = Float64MultiArray()
        msg.data = [0.0, 0.0, 0.0, 0.0]
        self.velocity_pub.publish(msg)
        self.motion_status_text.insert(tk.END, "Stop command sent: [0.0, 0.0, 0.0, 0.0]\n")
        self.motion_status_text.see(tk.END)
        self.update_status("Stopped")
        
    def objects_callback(self, msg):
        """Callback for objects_poses topic"""
        with self.objects_lock:
            # Store the latest objects data
            self.objects_data = {}
            for obj in msg.objects:
                self.objects_data[obj.object_name] = {
                    'position': [obj.pose.position.x, obj.pose.position.y, obj.pose.position.z],
                    'orientation': [obj.pose.orientation.x, obj.pose.orientation.y, obj.pose.orientation.z, obj.pose.orientation.w],
                    'roll': obj.roll,
                    'pitch': obj.pitch,
                    'yaw': obj.yaw,
                    'header': obj.header
                }
    
    def force_r2_callback(self, msg):
        """Callback for gripper R2 force topic"""
        with self.force_lock:
            self.gripper_force_r2 = abs(msg.data)  # Use absolute value for force
    
    def force_l2_callback(self, msg):
        """Callback for gripper L2 force topic"""
        with self.force_lock:
            self.gripper_force_l2 = abs(msg.data)  # Use absolute value for force
    
    def get_gripper_forces(self):
        """Get current gripper forces (thread-safe)"""
        with self.force_lock:
            return self.gripper_force_r2, self.gripper_force_l2
    
    def get_object_position(self, object_name):
        """Get position of a specific object by name"""
        with self.objects_lock:
            if object_name in self.objects_data:
                return self.objects_data[object_name]['position']
            return None
    
    def list_available_objects(self):
        """Get list of available object names"""
        with self.objects_lock:
            return list(self.objects_data.keys())
    
    def move_to_object(self):
        """Move arm to detected object position"""
        try:
            object_name = self.object_name_var.get()
            
            # Get object position
            position = self.get_object_position(object_name)
            
            if position is None:
                self.error_text.insert(tk.END, f"Error: Object '{object_name}' not found\n")
                self.error_text.insert(tk.END, f"Available objects: {', '.join(self.list_available_objects())}\n")
                self.error_text.see(tk.END)
                self.update_status(f"Object '{object_name}' not found")
                return
            
            # Convert from meters to millimeters
            x_mm = position[0] * 1000
            y_mm = position[1] * 1000
            z_mm = position[2] * 1000
            
            # Update coordinate fields
            self.x_var.set(f"{x_mm:.1f}")
            self.y_var.set(f"{y_mm:.1f}")
            self.z_var.set(f"{z_mm:.1f}")
            
            # Use existing move_to_position function
            self.error_text.insert(tk.END, f"Moving to object '{object_name}' at position: X={x_mm:.1f}mm, Y={y_mm:.1f}mm, Z={z_mm:.1f}mm\n")
            self.error_text.see(tk.END)
            self.update_status(f"Moving to object '{object_name}'")
            
            # Call the existing move_to_position function
            self.move_to_position()
            
        except Exception as e:
            self.error_text.insert(tk.END, f"Error moving to object: {str(e)}\n")
            self.error_text.see(tk.END)
            self.update_status(f"Error: {str(e)}")
    
    def refresh_objects(self):
        """Refresh and display available objects"""
        try:
            available_objects = self.list_available_objects()
            
            if available_objects:
                self.error_text.insert(tk.END, f"Available objects: {', '.join(available_objects)}\n")
                self.error_text.see(tk.END)
                self.update_status(f"Found {len(available_objects)} objects")
            else:
                self.error_text.insert(tk.END, "No objects detected. Make sure the object detection system is running.\n")
                self.error_text.see(tk.END)
                self.update_status("No objects detected")
                
        except Exception as e:
            self.error_text.insert(tk.END, f"Error refreshing objects: {str(e)}\n")
            self.error_text.see(tk.END)
            self.update_status(f"Error: {str(e)}")
    
    def update_status(self, message):
        """Update status label"""
        if hasattr(self, 'status_label'):
            self.status_label.config(text=message)
        
    def publish_joint_states(self):
        """Publish current joint states"""
        if self.running:
            self.joint_state.header.stamp = self.get_clock().now().to_msg()
            self.joint_state_pub.publish(self.joint_state)

def main(args=None):
    rclpy.init(args=args)
    
    gui = JETANKGripperControlGUI()
    
    try:
        rclpy.spin(gui)
    except KeyboardInterrupt:
        pass
    finally:
        gui.running = False
        gui.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

