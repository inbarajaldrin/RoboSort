#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import tkinter as tk
from tkinter import ttk
import threading
import time
import subprocess
import sys
import os
import numpy as np
import math

# Add scripts directory to path and import IK functions
scripts_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts')
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

try:
    from ik import compute_ik, forward_kinematics, verify_solution
except ImportError as e:
    # Try absolute path if relative path fails (for ros2 launch)
    scripts_path_abs = '/home/ubuntu/ros2_ws/src/JETANK_description/scripts'
    if scripts_path_abs not in sys.path:
        sys.path.insert(0, scripts_path_abs)
    try:
        from ik import compute_ik, forward_kinematics, verify_solution
    except ImportError as e2:
        print(f"Warning: Could not import IK functions: {e2}")
        print("IK functionality will be disabled.")
        compute_ik = None
        forward_kinematics = None
        verify_solution = None

class JETANKGripperControlGUI(Node):
    def __init__(self):
        super().__init__('jetank_gripper_control_gui')
        
        # Create publisher for joint states
        self.joint_state_pub = self.create_publisher(JointState, 'joint_states', 10)
        
        # Create trajectory publisher
        self.trajectory_pub = self.create_publisher(JointTrajectory, 'arm_trajectory', 10)
        
        # Initialize joint state message with all JETANK revolute joints
        self.joint_state = JointState()
        self.joint_state.header.frame_id = ''
        self.joint_state.name = [
            'revolute_BEARING',                  # Arm base rotation: -1.5708 to 1.5708
            'revolute_FREE_WHEEL_LEFT',          # Left free wheel: 0.0 to 6.283185
            'revolute_FREE_WHEEL_RIGHT',         # Right free wheel: 0.0 to 6.283185
            'revolute_GRIPPER_L1',               # Left gripper L1: -0.785398 to 0.0
            'revolute_GRIPPER_L2',               # Left gripper L2: -0.785398 to 0.0
            'Revolute_SERVO_UPPER',              # Upper arm servo: -3.1418 to 0.785594
            'Revolute_SERVO_LOWER',              # Lower arm servo: 0.0 to 1.570796
            'Revolute_DRIVING_WHEEL_R',          # Right driving wheel: 0.0 to 6.283185
            'Revolute_DRIVING_WHEEL_L',          # Left driving wheel: 0.0 to 6.283185
            'Revolute_GRIPPER_R2',               # Right gripper R2: -0.785398 to 0.0
            'Revolute_GRIPPER_R1',               # Right gripper R1: 0.0 to 0.785398
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
        self.bearing_scale = ttk.Scale(arm_frame, from_=-1.5708, to=1.5708,
                                      variable=self.bearing_var, orient=tk.HORIZONTAL,
                                      command=self.on_bearing_change)
        self.bearing_scale.pack(fill=tk.X, pady=2)
        self.bearing_label = ttk.Label(arm_frame, text="0.000")
        self.bearing_label.pack(anchor=tk.W)
        
        # Lower Servo
        ttk.Label(arm_frame, text="Lower Servo:").pack(anchor=tk.W)
        self.servo_lower_var = tk.DoubleVar(value=0.0)
        self.servo_lower_scale = ttk.Scale(arm_frame, from_=0.0, to=1.570796,
                                          variable=self.servo_lower_var, orient=tk.HORIZONTAL,
                                          command=self.on_servo_lower_change)
        self.servo_lower_scale.pack(fill=tk.X, pady=2)
        self.servo_lower_label = ttk.Label(arm_frame, text="0.000")
        self.servo_lower_label.pack(anchor=tk.W)
        
        # Upper Servo
        ttk.Label(arm_frame, text="Upper Servo:").pack(anchor=tk.W)
        self.servo_upper_var = tk.DoubleVar(value=0.0)
        self.servo_upper_scale = ttk.Scale(arm_frame, from_=-3.1418, to=0.785594,
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
        self.camera_tilt_scale = ttk.Scale(camera_frame, from_=-0.785398, to=0.785398,
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
        self.left_l1_scale = ttk.Scale(left_frame, from_=-0.785398, to=0.0, 
                                      variable=self.left_l1_var, orient=tk.HORIZONTAL,
                                      command=self.on_left_l1_change)
        self.left_l1_scale.pack(fill=tk.X, pady=2)
        self.left_l1_label = ttk.Label(left_frame, text="0.000")
        self.left_l1_label.pack(anchor=tk.W)
        
        ttk.Label(left_frame, text="L2 Finger:").pack(anchor=tk.W)
        self.left_l2_var = tk.DoubleVar(value=0.0)
        self.left_l2_scale = ttk.Scale(left_frame, from_=-0.785398, to=0.0,
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
        self.right_r1_scale = ttk.Scale(right_frame, from_=0.0, to=0.785398,
                                       variable=self.right_r1_var, orient=tk.HORIZONTAL,
                                       command=self.on_right_r1_change)
        self.right_r1_scale.pack(fill=tk.X, pady=2)
        self.right_r1_label = ttk.Label(right_frame, text="0.000")
        self.right_r1_label.pack(anchor=tk.W)
        
        ttk.Label(right_frame, text="R2 Finger:").pack(anchor=tk.W)
        self.right_r2_var = tk.DoubleVar(value=0.0)
        self.right_r2_scale = ttk.Scale(right_frame, from_=-0.785398, to=0.0,
                                       variable=self.right_r2_var, orient=tk.HORIZONTAL,
                                       command=self.on_right_r2_change)
        self.right_r2_scale.pack(fill=tk.X, pady=2)
        self.right_r2_label = ttk.Label(right_frame, text="0.000")
        self.right_r2_label.pack(anchor=tk.W)
        
        # Control buttons for individual tab
        button_frame = ttk.Frame(individual_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Open Gripper", 
                  command=self.open_gripper).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Close Gripper", 
                  command=self.close_gripper).pack(side=tk.LEFT, padx=5)
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
        
        # Coordinate input frame
        coord_frame = ttk.LabelFrame(arm_control_frame, text="Target Position", padding="10")
        coord_frame.pack(fill=tk.X, pady=5)
        
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
        
        # Duration input
        duration_frame = ttk.Frame(arm_control_frame)
        duration_frame.pack(fill=tk.X, pady=5)
        ttk.Label(duration_frame, text="Duration (sec):").pack(side=tk.LEFT)
        self.duration_var = tk.StringVar(value="3.0")
        self.duration_entry = ttk.Entry(duration_frame, textvariable=self.duration_var, width=10)
        self.duration_entry.pack(side=tk.LEFT, padx=(5, 0))
        
        # Move button
        button_frame = ttk.Frame(arm_control_frame)
        button_frame.pack(fill=tk.X, pady=10)
        ttk.Button(button_frame, text="Move to Position", 
                  command=self.move_to_position).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Move with Trajectory", 
                  command=self.move_with_trajectory).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Stop Trajectory", 
                  command=self.stop_trajectory).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset Position", 
                  command=self.reset_arm_position).pack(side=tk.LEFT, padx=5)
        
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
        self.error_text.insert(tk.END, "Enter target coordinates (in mm) and click 'Move to Position' for instant movement\n")
        self.error_text.insert(tk.END, "Or click 'Move with Trajectory' for smooth timed movement\n")
        self.error_text.insert(tk.END, "Inverse Kinematics will calculate the required joint angles\n\n")
        self.error_text.insert(tk.END, "Workspace limits (approximate):\n")
        self.error_text.insert(tk.END, "  X: -200 to 200 mm\n")
        self.error_text.insert(tk.END, "  Y: -200 to 200 mm\n")
        self.error_text.insert(tk.END, "  Z: 50 to 250 mm\n\n")
        
    def create_gripper_control_tab(self):
        """Create the gripper control tab"""
        gripper_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(gripper_frame, text="Gripper Control")
        
        # Left Side Control
        left_simple_frame = ttk.LabelFrame(gripper_frame, text="Left Side", padding="10")
        left_simple_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(left_simple_frame, text="Left Gripper:").pack(anchor=tk.W)
        self.simple_left_var = tk.DoubleVar(value=0.0)
        self.simple_left_scale = ttk.Scale(left_simple_frame, from_=0.0, to=-0.785398,
                                          variable=self.simple_left_var, orient=tk.HORIZONTAL,
                                          command=self.on_simple_left_change)
        self.simple_left_scale.pack(fill=tk.X, pady=2)
        self.simple_left_label = ttk.Label(left_simple_frame, text="0.000 (Closed)")
        self.simple_left_label.pack(anchor=tk.W)
        
        # Right Side Control
        right_simple_frame = ttk.LabelFrame(gripper_frame, text="Right Side", padding="10")
        right_simple_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(right_simple_frame, text="Right Gripper:").pack(anchor=tk.W)
        self.simple_right_var = tk.DoubleVar(value=0.0)
        self.simple_right_scale = ttk.Scale(right_simple_frame, from_=0.0, to=0.785398,
                                           variable=self.simple_right_var, orient=tk.HORIZONTAL,
                                           command=self.on_simple_right_change)
        self.simple_right_scale.pack(fill=tk.X, pady=2)
        self.simple_right_label = ttk.Label(right_simple_frame, text="0.000 (Closed)")
        self.simple_right_label.pack(anchor=tk.W)
        
        # Simple control buttons
        simple_button_frame = ttk.Frame(gripper_frame)
        simple_button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(simple_button_frame, text="Open Gripper", 
                  command=self.simple_open_gripper).pack(side=tk.LEFT, padx=5)
        ttk.Button(simple_button_frame, text="Close Gripper", 
                  command=self.simple_close_gripper).pack(side=tk.LEFT, padx=5)
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
        self.x_var.set("0.0")
        self.y_var.set("0.0")
        self.z_var.set("0.0")
        
        # Reset arm joints to default positions
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
        
        self.error_text.insert(tk.END, "Arm and camera reset to home position\n")
        self.error_text.see(tk.END)
        self.update_status("Arm and camera reset to home position")
        
    def move_with_trajectory(self):
        """Move arm to specified x, y, z position using inverse kinematics with timed trajectory"""
        try:
            x = float(self.x_var.get())
            y = float(self.y_var.get())
            z = float(self.z_var.get())
            duration = float(self.duration_var.get())
            
            # Check if IK functions are available
            if compute_ik is None:
                self.error_text.insert(tk.END, "Error: IK functions not available. Check ik.py import.\n")
                self.error_text.see(tk.END)
                self.update_status("Error: IK functions unavailable")
                return
            
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
            
        except ValueError:
            self.error_text.insert(tk.END, "Error: Please enter valid numeric values for coordinates and duration\n")
            self.error_text.see(tk.END)
            self.update_status("Error: Invalid input values")
        except Exception as e:
            self.error_text.insert(tk.END, f"Error: {str(e)}\n")
            self.error_text.see(tk.END)
            self.update_status(f"Error: {str(e)}")
            
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
        
    def on_simple_left_change(self, value):
        """Handle simple left slider change (controls both L1 and L2)"""
        pos = float(value)
        # Map slider value to joint positions: L1 and L2 both move together
        # Slider: 0.0 (closed) to -0.785 (open)
        self.left_l1_var.set(pos)
        self.left_l2_var.set(pos)
        self.joint_state.position[self.gripper_joint_indices['L1']] = pos  # L1
        self.joint_state.position[self.gripper_joint_indices['L2']] = pos  # L2
        
        # Update labels
        status = "Open" if pos < -0.1 else "Closed"
        self.simple_left_label.config(text=f"{pos:.3f} ({status})")
        if hasattr(self, 'left_l1_label'):
            self.left_l1_label.config(text=f"{pos:.3f}")
            self.left_l2_label.config(text=f"{pos:.3f}")
        self.update_status(f"Left side: {pos:.3f}")
        
    def on_simple_right_change(self, value):
        """Handle simple right slider change (controls both R1 and R2)"""
        pos = float(value)
        # Map slider value to joint positions according to your mapping:
        # R1: 0 to 0, 0.785 to 0.785 (direct mapping)
        # R2: 0 to 0, 0.785 to -0.785 (inverted mapping)
        
        # Calculate mapped positions
        r1_pos = pos   # R1: direct mapping (0.785 → 0.785)
        r2_pos = -pos  # R2: inverted mapping (0.785 → -0.785)
        
        self.right_r1_var.set(pos)      # R1 shows direct value
        self.right_r2_var.set(r2_pos)   # R2 shows inverted value
        
        self.joint_state.position[self.gripper_joint_indices['R1']] = r1_pos  # R1 (direct)
        self.joint_state.position[self.gripper_joint_indices['R2']] = r2_pos  # R2 (inverted)
        
        # Update labels
        status = "Open" if pos > 0.1 else "Closed"
        self.simple_right_label.config(text=f"{pos:.3f} ({status})")
        if hasattr(self, 'right_r1_label'):
            self.right_r1_label.config(text=f"{r1_pos:.3f}")
            self.right_r2_label.config(text=f"{r2_pos:.3f}")
        self.update_status(f"Right side: {pos:.3f}")
        
    def simple_open_gripper(self):
        """Open gripper using simple controls"""
        # Set left slider to open position
        self.simple_left_var.set(-0.785)
        # Set right slider to open position  
        self.simple_right_var.set(0.785)
        
        # Update joint positions with proper mappings
        self.joint_state.position[self.gripper_joint_indices['L1']] = -0.785  # L1
        self.joint_state.position[self.gripper_joint_indices['L2']] = -0.785  # L2
        self.joint_state.position[self.gripper_joint_indices['R1']] = 0.785   # R1 (direct mapping)
        self.joint_state.position[self.gripper_joint_indices['R2']] = -0.785  # R2 (inverted mapping from 0.785)
        
        # Update individual tab sliders if they exist
        if hasattr(self, 'left_l1_var'):
            self.left_l1_var.set(-0.785)
            self.left_l2_var.set(-0.785)
            self.right_r1_var.set(0.785)
            self.right_r2_var.set(-0.785)
            
        # Update labels
        self.simple_left_label.config(text="-0.785 (Open)")
        self.simple_right_label.config(text="0.785 (Open)")
        
        self.update_status("Gripper opened (simple control)")
        
    def simple_close_gripper(self):
        """Close gripper using simple controls"""
        # Set both sliders to closed position
        self.simple_left_var.set(0.0)
        self.simple_right_var.set(0.0)
        
        # Update joint positions
        self.joint_state.position[self.gripper_joint_indices['L1']] = 0.0  # L1
        self.joint_state.position[self.gripper_joint_indices['L2']] = 0.0  # L2
        self.joint_state.position[self.gripper_joint_indices['R1']] = 0.0  # R1
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
        
    def open_gripper(self):
        """Open gripper (move to open position)"""
        # Left gripper: move to more negative values (open)
        self.left_l1_var.set(-0.5)
        self.left_l2_var.set(-0.5)
        # Right gripper: move to more positive values (open)
        self.right_r1_var.set(0.5)
        self.right_r2_var.set(-0.5)
        
        # Update joint positions (only gripper joints, keep others at 0.0)
        self.joint_state.position[self.gripper_joint_indices['L1']] = -0.5
        self.joint_state.position[self.gripper_joint_indices['L2']] = -0.5
        self.joint_state.position[self.gripper_joint_indices['R1']] = 0.5
        self.joint_state.position[self.gripper_joint_indices['R2']] = -0.5
        
        # Update labels
        self.left_l1_label.config(text="-0.500")
        self.left_l2_label.config(text="-0.500")
        self.right_r1_label.config(text="0.500")
        self.right_r2_label.config(text="-0.500")
        
        self.update_status("Gripper opened")
        
    def close_gripper(self):
        """Close gripper (move to closed position)"""
        # Move all joints to closed position
        self.left_l1_var.set(0.0)
        self.left_l2_var.set(0.0)
        self.right_r1_var.set(0.0)
        self.right_r2_var.set(0.0)
        
        # Update joint positions (only gripper joints, keep others at 0.0)
        self.joint_state.position[self.gripper_joint_indices['L1']] = 0.0
        self.joint_state.position[self.gripper_joint_indices['L2']] = 0.0
        self.joint_state.position[self.gripper_joint_indices['R1']] = 0.0
        self.joint_state.position[self.gripper_joint_indices['R2']] = 0.0
        
        # Update labels
        self.left_l1_label.config(text="0.000")
        self.left_l2_label.config(text="0.000")
        self.right_r1_label.config(text="0.000")
        self.right_r2_label.config(text="0.000")
        
        self.update_status("Gripper closed")
        
    def reset_gripper(self):
        """Reset gripper and arm to default position"""
        self.close_gripper()
        
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

