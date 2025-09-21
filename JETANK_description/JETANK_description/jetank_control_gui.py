#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import tkinter as tk
from tkinter import ttk
import threading
import time

class JETANKGripperControlGUI(Node):
    def __init__(self):
        super().__init__('jetank_gripper_control_gui')
        
        # Create publisher for joint states
        self.joint_state_pub = self.create_publisher(JointState, 'joint_states', 10)
        
        # Initialize joint state message with all JETANK revolute joints
        self.joint_state = JointState()
        self.joint_state.header.frame_id = ''
        self.joint_state.name = [
            'revolute_BEARING',           # Arm base rotation: -1.5708 to 1.5708
            'revolute_FREE_WHEEL_LEFT',   # Left free wheel: 0.0 to 6.283185
            'revolute_FREE_WHEEL_RIGHT',  # Right free wheel: 0.0 to 6.283185
            'revolute_GRIPPER_L1',        # Left gripper L1: -0.785398 to 0.0
            'revolute_GRIPPER_L2',        # Left gripper L2: -0.785398 to 0.0
            'Revolute_SERVO_UPPER',       # Upper arm servo: -3.1418 to 0.785594
            'Revolute_SERVO_LOWER',       # Lower arm servo: 0.0 to 1.570796
            'Revolute_DRIVING_WHEEL_R',   # Right driving wheel: 0.0 to 6.283185
            'Revolute_DRIVING_WHEEL_L',   # Left driving wheel: 0.0 to 6.283185
            'Revolute_GRIPPER_R2',        # Right gripper R2: -0.785398 to 0.0
            'Revolute_GRIPPER_R1'         # Right gripper R1: 0.0 to 0.785398
        ]
        # Initialize all joints to default positions (0.0 for most, except wheels which can be 0.0)
        self.joint_state.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.joint_state.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.joint_state.effort = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
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
        
        # Threading control
        self.gui_thread = None
        self.running = True
        
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
        self.root.geometry("450x500")
        
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
        
        # Move button
        button_frame = ttk.Frame(arm_control_frame)
        button_frame.pack(fill=tk.X, pady=10)
        ttk.Button(button_frame, text="Move to Position", 
                  command=self.move_to_position).pack(side=tk.LEFT, padx=5)
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
        self.error_text.insert(tk.END, "Arm Control Ready\nEnter target coordinates and click 'Move to Position'\n")
        
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
        
    def move_to_position(self):
        """Move arm to specified x, y, z position"""
        try:
            x = float(self.x_var.get())
            y = float(self.y_var.get())
            z = float(self.z_var.get())
            
            # TODO: Add inverse kinematics logic here
            self.error_text.insert(tk.END, f"Moving to position: X={x}, Y={y}, Z={z}\n")
            self.error_text.see(tk.END)
            self.update_status(f"Moving to ({x}, {y}, {z})")
            
        except ValueError:
            self.error_text.insert(tk.END, "Error: Please enter valid numeric values for coordinates\n")
            self.error_text.see(tk.END)
            self.update_status("Error: Invalid coordinates")
            
    def reset_arm_position(self):
        """Reset arm to home position"""
        self.x_var.set("0.0")
        self.y_var.set("0.0")
        self.z_var.set("0.0")
        
        # Reset arm joints to default positions
        self.bearing_var.set(0.0)
        self.servo_lower_var.set(0.0)
        self.servo_upper_var.set(0.0)
        
        # Update joint positions
        self.joint_state.position[self.arm_joint_indices['BEARING']] = 0.0
        self.joint_state.position[self.arm_joint_indices['SERVO_LOWER']] = 0.0
        self.joint_state.position[self.arm_joint_indices['SERVO_UPPER']] = 0.0
        
        # Update labels
        self.bearing_label.config(text="0.000")
        self.servo_lower_label.config(text="0.000")
        self.servo_upper_label.config(text="0.000")
        
        self.error_text.insert(tk.END, "Arm reset to home position\n")
        self.error_text.see(tk.END)
        self.update_status("Arm reset to home position")
        
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
        
        # Update joint positions
        self.joint_state.position[self.arm_joint_indices['BEARING']] = 0.0
        self.joint_state.position[self.arm_joint_indices['SERVO_LOWER']] = 0.0
        self.joint_state.position[self.arm_joint_indices['SERVO_UPPER']] = 0.0
        
        # Update labels
        self.bearing_label.config(text="0.000")
        self.servo_lower_label.config(text="0.000")
        self.servo_upper_label.config(text="0.000")
        
        self.update_status("Gripper and arm reset")
        
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
