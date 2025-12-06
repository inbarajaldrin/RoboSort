#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from std_msgs.msg import Float64MultiArray, Float32
from geometry_msgs.msg import Twist
from rosgraph_msgs.msg import Clock
from tf2_msgs.msg import TFMessage
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

# Try to import scipy - required for IK functions
try:
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation as R
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    minimize = None
    R = None
    print("Warning: scipy not available. IK functions will be disabled.")

# IK functions directly embedded in this file
def forward_kinematics(theta0, theta1, theta3):
    """
    Calculate forward kinematics for JETANK robot
    Returns transformation matrix and position
    """
    # Joint limits (from URDF)
    limits = [
        {'lower': -1.5708, 'upper': 1.5708},      # Joint 0
        {'lower': 0.0,     'upper': 1.570796},    # Joint 1
        {'lower': -3.1418, 'upper': 0.785594}     # Joint 3
    ]
    
    # Check joint limits
    if theta0 < limits[0]['lower'] or theta0 > limits[0]['upper']:
        return None, None
    if theta1 < limits[1]['lower'] or theta1 > limits[1]['upper']:
        return None, None
    if theta3 < limits[2]['lower'] or theta3 > limits[2]['upper']:
        return None, None
    
    # DH parameters (in meters)
    dh_params = [
        {'alpha': 1.57, 'a': -13.50/1000, 'd': 75.00/1000},  # Link 0
        {'alpha': -1.57,  'a': 0.00/1000,   'd': 0.00/1000},   # Link 1  
        {'alpha': 1.57, 'a': 0.00/1000,   'd': 95.00/1000},  # Link 2
        {'alpha': -1.57,  'a': -179.25/1000, 'd': 0.00/1000}   # Link 3
    ]
    
    # Joint angles (theta2 is fixed at 0)
    theta = [theta0, theta1, 0.0, theta3]
    
    def dh_matrix(alpha, a, d, theta):
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        cos_alpha = math.cos(alpha)
        sin_alpha = math.sin(alpha)
        
        return np.array([
            [cos_theta, -sin_theta*cos_alpha,  sin_theta*sin_alpha, a*cos_theta],
            [sin_theta,  cos_theta*cos_alpha, -cos_theta*sin_alpha, a*sin_theta],
            [0,          sin_alpha,            cos_alpha,           d],
            [0,          0,                    0,                   1]
        ])
    
    # Calculate transformation matrix
    T = np.eye(4)
    for i in range(4):
        dh = dh_params[i]
        T_i = dh_matrix(dh['alpha'], dh['a'], dh['d'], theta[i])
        T = np.dot(T, T_i)
    
    # Extract position and convert to mm
    position = T[:3, 3] * 1000
    
    return T, (position[0], position[1], position[2])

def rpy_to_matrix(rpy_degrees):
    """Convert roll-pitch-yaw in degrees to rotation matrix"""
    if not SCIPY_AVAILABLE:
        return None
    return R.from_euler('xyz', rpy_degrees, degrees=True).as_matrix()

def ik_objective(q, target_position, target_orientation=None, position_weight=1.0, orientation_weight=0.1):
    """
    Objective function for IK optimization
    q: joint angles [theta0, theta1, theta3]
    target_position: [x, y, z] in mm
    target_orientation: optional 3x3 rotation matrix
    """
    # Get forward kinematics
    T, current_pos = forward_kinematics(q[0], q[1], q[2])
    
    if T is None or current_pos is None:
        return 1e6  # Large penalty for invalid joint angles
    
    # Position error
    pos_error = np.linalg.norm(np.array(current_pos) - np.array(target_position))
    
    total_error = position_weight * pos_error
    
    # Orientation error (if specified)
    if target_orientation is not None:
        current_orientation = T[:3, :3]
        rot_error = np.linalg.norm(current_orientation - target_orientation, 'fro')
        total_error += orientation_weight * rot_error
    
    return total_error

def compute_ik(target_x, target_y, target_z, target_rpy=None, q_guess=None, max_tries=10, position_tolerance=1.0):
    """
    Compute inverse kinematics for JETANK robot
    
    Args:
        target_x, target_y, target_z: target position in mm
        target_rpy: optional target orientation [rx, ry, rz] in degrees
        q_guess: initial guess for joint angles
        max_tries: maximum number of optimization attempts
        position_tolerance: acceptable position error in mm
    
    Returns:
        joint_angles: [theta0, theta1, theta3] in radians, or None if failed
    """
    if not SCIPY_AVAILABLE:
        return None
    
    target_position = [target_x, target_y, target_z]
    target_orientation = None
    
    if target_rpy is not None:
        target_orientation = rpy_to_matrix(target_rpy)
    
    # Joint bounds
    joint_bounds = [
        (-1.5708, 1.5708),   # theta0
        (0.0, 1.570796),     # theta1  
        (-3.1418, 0.785594)  # theta3
    ]
    
    # Default initial guess if none provided
    if q_guess is None:
        q_guess = [0.0, 0.785, -1.57]  # Middle-ish values
    
    best_result = None
    best_error = float('inf')
    
    for attempt in range(max_tries):
        # Add small random perturbation to initial guess for each attempt
        if attempt > 0:
            current_guess = q_guess + np.random.normal(0, 0.1, 3)
            # Ensure guess is within bounds
            for i in range(3):
                current_guess[i] = np.clip(current_guess[i], joint_bounds[i][0], joint_bounds[i][1])
        else:
            current_guess = q_guess
        
        try:
            # Use different optimization methods
            methods = ['L-BFGS-B', 'SLSQP', 'trust-constr']
            method = methods[attempt % len(methods)]
            
            result = minimize(
                ik_objective, 
                current_guess, 
                args=(target_position, target_orientation),
                method=method,
                bounds=joint_bounds,
                options={'maxiter': 1000}
            )
            
            if result.success:
                error = ik_objective(result.x, target_position, target_orientation)
                if error < best_error:
                    best_error = error
                    best_result = result
                
                # Check if solution is good enough
                if error < position_tolerance:
                    print(f"IK converged on attempt {attempt + 1}")
                    print(f"Position error: {error:.3f} mm")
                    return result.x
        
        except Exception as e:
            continue
    
    # Return best result found, even if not optimal
    if best_result is not None and best_error < 10.0:  # Accept if within 10mm
        print(f"IK converged with error: {best_error:.3f} mm")
        return best_result.x
    
    print(f"IK failed to converge after {max_tries} attempts")
    print(f"Best error achieved: {best_error:.3f} mm")
    return None

def verify_solution(joint_angles, target_position, target_rpy=None):
    """Verify the IK solution by computing forward kinematics"""
    if not SCIPY_AVAILABLE:
        return False
    
    T, actual_pos = forward_kinematics(joint_angles[0], joint_angles[1], joint_angles[2])
    
    if actual_pos is None:
        print("Invalid joint configuration!")
        return False
    
    pos_error = np.linalg.norm(np.array(actual_pos) - np.array(target_position))
    
    print(f"\nVerification:")
    print(f"Target position: [{target_position[0]:.2f}, {target_position[1]:.2f}, {target_position[2]:.2f}] mm")
    print(f"Actual position: [{actual_pos[0]:.2f}, {actual_pos[1]:.2f}, {actual_pos[2]:.2f}] mm")
    print(f"Position error: {pos_error:.3f} mm")
    
    if target_rpy is not None:
        actual_rpy = R.from_matrix(T[:3, :3]).as_euler('xyz', degrees=True)
        print(f"Target orientation: [{target_rpy[0]:.1f}, {target_rpy[1]:.1f}, {target_rpy[2]:.1f}] deg")
        print(f"Actual orientation: [{actual_rpy[0]:.1f}, {actual_rpy[1]:.1f}, {actual_rpy[2]:.1f}] deg")
    
    return pos_error < 5.0  # Accept if within 5mm

# Set IK functions to None if scipy is not available (so code can check if compute_ik is None)
if not SCIPY_AVAILABLE:
    compute_ik = None
    verify_solution = None

class JETANKGripperControlGUI(Node):
    def __init__(self):
        super().__init__('jetank_gripper_control_gui')
        
        # Hardware mode: False = Fake Hardware (simulation), True = Real Hardware
        self.use_real_hardware = False
        self.hardware_mode_lock = threading.Lock()
        
        # Wheel inversion compensation: True = wheels are inverted (need to negate commands)
        # Set to False if wheels are correctly wired
        self.wheels_inverted = True
        
        # Real hardware joint mapping (5 real joints -> 12 simulated joints)
        # Real robot joints: base_joint, shoulder_joint, elbow_joint, wrist_joint, camera_joint
        self.real_joint_names = ['base_joint', 'shoulder_joint', 'elbow_joint', 'wrist_joint', 'camera_joint']
        
        # Mapping: real joint name -> simulated joint index
        self.real_to_sim_mapping = {
            'base_joint': 0,       # -> revolute_BEARING
            'shoulder_joint': 6,   # -> Revolute_SERVO_LOWER
            'elbow_joint': 5,      # -> Revolute_SERVO_UPPER
            'wrist_joint': None,   # -> Controls all 4 gripper joints (L1, L2, R1, R2)
            'camera_joint': 11     # -> revolute_CAMERA_HOLDER_ARM_LOWER
        }
        
        # Gripper indices controlled by wrist_joint
        self.wrist_to_gripper_indices = [3, 4, 9, 10]  # L1, L2, R2, R1
        
        # Real hardware state (from servo driver)
        self.real_joint_state = None
        self.real_joint_state_lock = threading.Lock()
        
        # Load joint limits from URDF
        self.joint_limits = self.load_joint_limits_from_urdf()
        
        # Create publisher for joint states (for RViz visualization)
        self.joint_state_pub = self.create_publisher(JointState, 'joint_states', 10)
        
        # Create publisher for real hardware joint commands
        self.joint_command_pub = self.create_publisher(JointState, 'joint_commands', 10)
        
        # Create trajectory publisher
        self.trajectory_pub = self.create_publisher(JointTrajectory, 'arm_trajectory', 10)
        
        # Create velocity controller publisher (for fake hardware/simulation)
        self.velocity_pub = self.create_publisher(Float64MultiArray, '/forward_velocity_controller/commands', 10)
        
        # Create cmd_vel publisher (for real hardware motor driver)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Subscriber for real hardware joint states (from servo driver)
        # Will be created/destroyed when switching modes
        self.real_joint_state_sub = None
        
        # Subscriber for joint commands (works in both simulation and real mode)
        # Allows external nodes to control the robot by publishing to joint_commands
        self.joint_command_sub = self.create_subscription(
            JointState,
            'joint_commands',
            self.joint_command_callback,
            10
        )
        
        # Object detection data
        self.objects_data = {}  # Store latest objects data
        self.objects_lock = threading.Lock()  # Thread safety for objects data
        
        # Create subscriber for object poses (will be initialized after GUI setup)
        # Start with default topic, can be updated via GUI
        self.objects_sub = None
        self.objects_topic = '/objects_poses'  # Default topic name
        self._create_objects_subscription()
        
        # Force monitoring for gripper
        self.gripper_force_r2 = 0.0
        self.gripper_force_l2 = 0.0
        self.force_lock = threading.Lock()
        self.force_threshold = 0.5  # Default threshold (stops when both R2 and L2 exceed this value)
        
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
        
        # Simulation clock synchronization (only active during gripper closing)
        self.sim_clock = None
        self.sim_clock_lock = threading.Lock()
        self.sim_clock_sub = None  # Will be created when closing gripper
        self.sim_time_offset = None  # Initial sim time when closing starts
        self.use_sim_clock = False  # Flag to enable sim clock usage
        self.publish_rate = 10.0  # Target publishing rate (Hz)
        self.last_publish_sim_time = None
        
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
        
        # Gripper retry control
        # We retry up to 3 times after reaching the force threshold to handle potential
        # slip in simulation and ensure the object is securely held by the gripper
        self.gripper_retry_count = 0
        self.max_gripper_retries = 3
        self.force_threshold_reached = False
        
        # Create Tkinter GUI in a separate thread
        self.setup_gui_thread()
        
        # Create timer for publishing joint states (will be adjusted based on sim clock)
        # Start with 0.1s (10Hz), will adjust dynamically
        self.timer = self.create_timer(0.1, self.publish_joint_states)
        
        self.get_logger().info('JETANK Control GUI initialized')
    
    # ==================== Real Hardware Methods ====================
    
    def set_hardware_mode(self, use_real):
        """Switch between fake and real hardware modes"""
        with self.hardware_mode_lock:
            if use_real == self.use_real_hardware:
                return  # No change needed
            
            self.use_real_hardware = use_real
            
            if use_real:
                # Switch to real hardware mode
                self.get_logger().info('Switching to REAL hardware mode')
                # Create subscriber for real joint states from servo driver
                if self.real_joint_state_sub is None:
                    self.real_joint_state_sub = self.create_subscription(
                        JointState,
                        'real_joint_states',  # From servo driver
                        self.real_joint_state_callback,
                        10
                    )
                self.update_status("Mode: Real Hardware")
            else:
                # Switch to fake hardware mode
                self.get_logger().info('Switching to FAKE hardware mode')
                # Destroy real joint state subscriber
                if self.real_joint_state_sub is not None:
                    self.destroy_subscription(self.real_joint_state_sub)
                    self.real_joint_state_sub = None
                self.update_status("Mode: Fake Hardware")
    
    def joint_command_callback(self, msg):
        """Callback for receiving joint commands (works in simulation mode only)
        
        Only supports 12 simulated joints format (URDF joint names).
        In real hardware mode, commands are handled by hardware, not simulation.
        """
        # In real hardware mode, commands go to hardware, not simulation
        if self.use_real_hardware:
            return  # Commands are handled by hardware, not simulation
        
        if not msg.name or not msg.position:
            return
        
        # Create mapping from joint name to position
        command_positions = {}
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                command_positions[name] = msg.position[i]
        
        # Update joint state positions - only support 12 simulated joints format
        positions = list(self.joint_state.position)
        
        # Map commands to simulated joints (must match URDF joint names)
        for i, name in enumerate(self.joint_state.name):
            if name in command_positions:
                positions[i] = command_positions[name]
                # Apply joint limits if available
                if name in self.joint_limits:
                    lower = self.joint_limits[name].get('lower', -float('inf'))
                    upper = self.joint_limits[name].get('upper', float('inf'))
                    positions[i] = max(lower, min(upper, positions[i]))
        
        # Update joint state
        self.joint_state.position = positions
        
        self.get_logger().debug(f'Updated joint states from external command: {len(msg.name)} simulated joints')
    
    def real_joint_state_callback(self, msg):
        """Callback for receiving joint states from real hardware (servo driver)"""
        with self.real_joint_state_lock:
            self.real_joint_state = msg
        
        # Map real joint states to simulated joint states for RViz
        if self.use_real_hardware:
            self.map_real_to_simulated_joints(msg)
    
    def map_real_to_simulated_joints(self, real_msg):
        """Map 5 real robot joints to 12 simulated joints for RViz visualization"""
        # Create mapping from joint name to position
        real_positions = {}
        for i, name in enumerate(real_msg.name):
            if i < len(real_msg.position):
                real_positions[name] = real_msg.position[i]
        
        # Map to simulated joints
        positions = list(self.joint_state.position)
        
        # Store real positions for GUI slider updates
        base_pos = None
        shoulder_pos = None
        elbow_pos = None
        camera_pos = None
        wrist_pos = None
        
        for real_name, sim_index in self.real_to_sim_mapping.items():
            if real_name in real_positions:
                if sim_index is not None:
                    # Direct mapping
                    positions[sim_index] = real_positions[real_name]
                    # Store for GUI updates
                    if real_name == 'base_joint':
                        base_pos = real_positions[real_name]
                    elif real_name == 'shoulder_joint':
                        shoulder_pos = real_positions[real_name]
                    elif real_name == 'elbow_joint':
                        elbow_pos = real_positions[real_name]
                    elif real_name == 'camera_joint':
                        camera_pos = real_positions[real_name]
                elif real_name == 'wrist_joint':
                    # Wrist joint controls all 4 gripper joints
                    wrist_pos = real_positions[real_name]
                    wrist_angle = real_positions[real_name]
                    # Map wrist angle to gripper positions
                    # L1 and L2 are negative, R1 is positive, R2 is negative
                    gripper_angle = self.wrist_to_gripper_angle(wrist_angle)
                    positions[3] = -abs(gripper_angle)  # L1 (negative)
                    positions[4] = -abs(gripper_angle)  # L2 (negative)
                    positions[9] = -abs(gripper_angle)  # R2 (negative)
                    positions[10] = abs(gripper_angle)  # R1 (positive)
        
        # Update joint state
        self.joint_state.position = positions
        
        # Update GUI sliders thread-safely (only in real mode)
        if self.use_real_hardware and hasattr(self, 'root') and self.root.winfo_exists():
            self.root.after(0, self.update_sliders_from_real_state, 
                          base_pos, shoulder_pos, elbow_pos, camera_pos, wrist_pos)
    
    def update_sliders_from_real_state(self, base_pos, shoulder_pos, elbow_pos, camera_pos, wrist_pos):
        """Update GUI sliders from real robot joint states (called from main thread)"""
        try:
            # Update arm sliders
            if base_pos is not None and hasattr(self, 'bearing_var'):
                self.bearing_var.set(base_pos)
                if hasattr(self, 'bearing_label'):
                    self.bearing_label.config(text=f"{base_pos:.3f}")
            
            if shoulder_pos is not None and hasattr(self, 'servo_lower_var'):
                self.servo_lower_var.set(shoulder_pos)
                if hasattr(self, 'servo_lower_label'):
                    self.servo_lower_label.config(text=f"{shoulder_pos:.3f}")
            
            if elbow_pos is not None and hasattr(self, 'servo_upper_var'):
                self.servo_upper_var.set(elbow_pos)
                if hasattr(self, 'servo_upper_label'):
                    self.servo_upper_label.config(text=f"{elbow_pos:.3f}")
            
            if camera_pos is not None and hasattr(self, 'camera_tilt_var'):
                self.camera_tilt_var.set(camera_pos)
                if hasattr(self, 'camera_tilt_label'):
                    degrees = math.degrees(camera_pos)
                    self.camera_tilt_label.config(text=f"{camera_pos:.3f} rad ({degrees:.1f}°)")
            
            # Update gripper slider from wrist position
            if wrist_pos is not None:
                # Convert wrist angle to gripper servo value (0.0 to 1.22)
                gripper_angle = self.wrist_to_gripper_angle(wrist_pos)
                max_angle = self.get_joint_limit('Revolute_GRIPPER_R1', 'upper')
                if max_angle > 0:
                    # Convert gripper angle to servo value: servo_value = (gripper_angle / max_angle) * 1.22
                    servo_value = (abs(gripper_angle) / max_angle) * 1.22
                    servo_value = max(0.0, min(1.22, servo_value))  # Clamp to [0, 1.22]
                    
                    if hasattr(self, 'gripper_var'):
                        self.gripper_var.set(servo_value)
                    if hasattr(self, 'gripper_label'):
                        status = "Open" if servo_value > 0.1 else "Closed"
                        self.gripper_label.config(text=f"{servo_value:.3f} ({status})")
                    
                    # Update individual gripper sliders if they exist
                    gripper_angle_abs = abs(gripper_angle)
                    if hasattr(self, 'left_l1_var'):
                        self.left_l1_var.set(-gripper_angle_abs)
                        self.left_l2_var.set(-gripper_angle_abs)
                    if hasattr(self, 'right_r1_var'):
                        self.right_r1_var.set(gripper_angle_abs)
                        self.right_r2_var.set(-gripper_angle_abs)
                    if hasattr(self, 'left_l1_label'):
                        self.left_l1_label.config(text=f"{-gripper_angle_abs:.3f}")
                        self.left_l2_label.config(text=f"{-gripper_angle_abs:.3f}")
                    if hasattr(self, 'right_r1_label'):
                        self.right_r1_label.config(text=f"{gripper_angle_abs:.3f}")
                        self.right_r2_label.config(text=f"{-gripper_angle_abs:.3f}")
        except Exception as e:
            # Silently ignore GUI update errors (sliders might not exist yet)
            pass
    
    def wrist_to_gripper_angle(self, wrist_angle):
        """Convert wrist servo angle to gripper finger angles"""
        # The wrist servo controls the gripper opening/closing
        # Map wrist angle range to gripper angle range
        # Actual wrist servo max range: 0.0 (closed) to 1.22 (fully open) radians
        gripper_max = 1.047198  # URDF gripper joint limit
        wrist_max = 1.22  # Actual maximum wrist servo angle (fully open)
        # Normalize wrist angle and scale to gripper range
        gripper_angle = wrist_angle * gripper_max / wrist_max
        return max(-gripper_max, min(gripper_max, gripper_angle))
    
    def gripper_to_wrist_angle(self, gripper_angle):
        """Convert gripper finger angle to wrist servo angle"""
        # Reverse of wrist_to_gripper_angle
        # Actual wrist servo max range: 0.0 (closed) to 1.22 (fully open) radians
        gripper_max = 1.047198  # URDF gripper joint limit
        wrist_max = 1.22  # Actual maximum wrist servo angle (fully open)
        wrist_angle = gripper_angle * wrist_max / gripper_max
        return wrist_angle
    
    def send_real_hardware_command(self, joint_name, position, velocity=None):
        """Send joint command to real hardware"""
        if not self.use_real_hardware:
            return
        
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [joint_name]
        msg.position = [position]
        if velocity is not None:
            msg.velocity = [velocity]
        
        self.joint_command_pub.publish(msg)
        self.get_logger().info(f'Sent command to {joint_name}: {position:.3f} rad')
    
    def send_arm_command_real(self, bearing, shoulder, elbow):
        """Send arm joint commands to real hardware (sends each joint separately)"""
        if not self.use_real_hardware:
            return
        
        # Send each joint separately (same as slider behavior)
        self.send_real_hardware_command('base_joint', bearing)
        self.send_real_hardware_command('shoulder_joint', shoulder)
        self.send_real_hardware_command('elbow_joint', elbow)
        
        self.get_logger().info(f'Sent arm command: base={bearing:.3f}, shoulder={shoulder:.3f}, elbow={elbow:.3f}')
    
    def send_gripper_command_real(self, wrist_angle):
        """Send gripper (wrist) command to real hardware"""
        if not self.use_real_hardware:
            return
        
        # Clamp wrist angle to hardware limits: [0.0 (closed) to 1.22 (fully open)]
        wrist_max = 1.22  # Maximum wrist servo angle (fully open)
        wrist_min = 0.014   # Minimum wrist servo angle (closed)
        wrist_angle = max(wrist_min, min(wrist_max, wrist_angle))
        
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['wrist_joint']
        msg.position = [wrist_angle]
        
        self.joint_command_pub.publish(msg)
        self.get_logger().info(f'Sent gripper command: wrist={wrist_angle:.3f}')
    
    def send_camera_command_real(self, camera_angle):
        """Send camera tilt command to real hardware"""
        if not self.use_real_hardware:
            return
        
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['camera_joint']
        msg.position = [camera_angle]
        
        self.joint_command_pub.publish(msg)
        self.get_logger().info(f'Sent camera command: {camera_angle:.3f}')
    
    # ==================== End Real Hardware Methods ====================
    
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
        self.root.title("JETANK Control")
        self.root.geometry("450x580")
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="JETANK Control", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 10))
        
        # Hardware Mode Selector
        hardware_frame = ttk.LabelFrame(main_frame, text="Hardware Mode", padding="5")
        hardware_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.hardware_mode_var = tk.StringVar(value="fake")
        
        fake_radio = ttk.Radiobutton(hardware_frame, text="Fake Hardware (Simulation)", 
                                     variable=self.hardware_mode_var, value="fake",
                                     command=self.on_hardware_mode_change)
        fake_radio.pack(side=tk.LEFT, padx=10)
        
        real_radio = ttk.Radiobutton(hardware_frame, text="Real Hardware", 
                                     variable=self.hardware_mode_var, value="real",
                                     command=self.on_hardware_mode_change)
        real_radio.pack(side=tk.LEFT, padx=10)
        
        # Hardware status indicator
        self.hardware_status_label = ttk.Label(hardware_frame, text="● Fake", 
                                              foreground="blue", font=('Arial', 9, 'bold'))
        self.hardware_status_label.pack(side=tk.RIGHT, padx=10)
        
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
        self.topic_var = tk.StringVar(value=self.objects_topic)
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
        self.move_to_object_button = ttk.Button(object_button_frame, text="Move to Grab", 
                  command=self.move_to_object)
        self.move_to_object_button.pack(side=tk.LEFT, padx=5)
        ttk.Button(object_button_frame, text="Refresh Objects", 
                  command=self.refresh_objects).pack(side=tk.LEFT, padx=5)
        ttk.Button(object_button_frame, text="Update Topic", 
                  command=self.update_topic).pack(side=tk.LEFT, padx=5)
        
        # Update button text based on current topic (after button is created)
        if hasattr(self, 'objects_topic'):
            self._update_button_text()
        
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
        self.error_text.insert(tk.END, "  Enter object name and click 'Move to Grab' (or 'Move to Drop' for drop_poses) to move to detected object\n")
        self.error_text.insert(tk.END, "  Click 'Refresh Objects' to see available objects\n")
        self.error_text.insert(tk.END, "  Object positions are automatically converted from meters to mm\n")
        self.error_text.insert(tk.END, "  For drop_poses topic, a +0.05m (+50mm) z-offset is automatically applied\n\n")
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
        # Only update internal state in fake hardware mode
        if not self.use_real_hardware:
            self.joint_state.position[self.arm_joint_indices['BEARING']] = pos
        else:
            # In real mode, only send command - visualization comes from real robot
            self.send_real_hardware_command('base_joint', pos)
        self.update_status(f"Base Bearing: {pos:.3f}")
        
    def on_servo_lower_change(self, value):
        """Handle lower servo slider change"""
        pos = float(value)
        self.servo_lower_label.config(text=f"{pos:.3f}")
        # Only update internal state in fake hardware mode
        if not self.use_real_hardware:
            self.joint_state.position[self.arm_joint_indices['SERVO_LOWER']] = pos
        else:
            # In real mode, only send command - visualization comes from real robot
            self.send_real_hardware_command('shoulder_joint', pos)
        self.update_status(f"Lower Servo: {pos:.3f}")
        
    def on_servo_upper_change(self, value):
        """Handle upper servo slider change"""
        pos = float(value)
        self.servo_upper_label.config(text=f"{pos:.3f}")
        # Only update internal state in fake hardware mode
        if not self.use_real_hardware:
            self.joint_state.position[self.arm_joint_indices['SERVO_UPPER']] = pos
        else:
            # In real mode, only send command - visualization comes from real robot
            self.send_real_hardware_command('elbow_joint', pos)
        self.update_status(f"Upper Servo: {pos:.3f}")
    
    def on_camera_tilt_change(self, value):
        """Handle camera tilt slider change"""
        pos = float(value)
        degrees = math.degrees(pos)
        self.camera_tilt_label.config(text=f"{pos:.3f} rad ({degrees:.1f}°)")
        # Only update internal state in fake hardware mode
        if not self.use_real_hardware:
            self.joint_state.position[self.camera_joint_indices['CAMERA_TILT']] = pos
        else:
            # In real mode, only send command - visualization comes from real robot
            self.send_real_hardware_command('camera_joint', pos)
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
                    
                    # Update labels
                    self.bearing_label.config(text=f"{theta0:.3f}")
                    self.servo_lower_label.config(text=f"{theta1:.3f}")
                    self.servo_upper_label.config(text=f"{theta3:.3f}")
                    
                    if not self.use_real_hardware:
                        # Fake mode: Update GUI sliders and joint positions
                        self.bearing_var.set(theta0)
                        self.servo_lower_var.set(theta1)
                        self.servo_upper_var.set(theta3)
                        
                        # Update joint state message
                        self.joint_state.position[self.arm_joint_indices['BEARING']] = theta0
                        self.joint_state.position[self.arm_joint_indices['SERVO_LOWER']] = theta1
                        self.joint_state.position[self.arm_joint_indices['SERVO_UPPER']] = theta3
                    else:
                        # Real mode: Only send commands, don't update joint_state
                        # Visualization comes from real robot feedback
                        self.send_arm_command_real(theta0, theta1, theta3)
                        # Update sliders for display only (callbacks won't update joint_state in real mode)
                        self.bearing_var.set(theta0)
                        self.servo_lower_var.set(theta1)
                        self.servo_upper_var.set(theta3)
                    
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
            # Update labels
            self.bearing_label.config(text="0.000")
            self.servo_lower_label.config(text="0.000")
            self.servo_upper_label.config(text="0.000")
            
            if not self.use_real_hardware:
                # Fake mode: Update GUI sliders and joint positions
                self.bearing_var.set(0.0)
                self.servo_lower_var.set(0.0)
                self.servo_upper_var.set(0.0)
                
                # Update joint positions (arm only)
                self.joint_state.position[self.arm_joint_indices['BEARING']] = 0.0
                self.joint_state.position[self.arm_joint_indices['SERVO_LOWER']] = 0.0
                self.joint_state.position[self.arm_joint_indices['SERVO_UPPER']] = 0.0
            else:
                # Real mode: Only send commands, don't update joint_state
                # Visualization comes from real robot feedback
                self.send_arm_command_real(0.0, 0.0, 0.0)
                # Update sliders for display only (callbacks won't update joint_state in real mode)
                self.bearing_var.set(0.0)
                self.servo_lower_var.set(0.0)
                self.servo_upper_var.set(0.0)
            
            self.error_text.insert(tk.END, "Arm reset to home position (camera unchanged)\n")
            self.error_text.see(tk.END)
            self.update_status("Arm reset to home position")
    
    def camera_down(self):
        """Move camera down to -45 degrees"""
        # Convert -45 degrees to radians
        target_angle = math.radians(-45.0)
        
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
                self.error_text.insert(tk.END, f"  To: -45.0°\n")
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
            self.camera_tilt_label.config(text=f"{target_angle:.3f} rad (-45.0°)")
            
            # Only update internal state in fake hardware mode
            if not self.use_real_hardware:
                self.joint_state.position[self.camera_joint_indices['CAMERA_TILT']] = target_angle
            else:
                # In real mode, send command to camera
                self.send_real_hardware_command('camera_joint', target_angle)
            
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
            self.camera_tilt_label.config(text=f"{target_angle:.3f} rad (0.0°)")
            
            # Only update internal state in fake hardware mode
            if not self.use_real_hardware:
                self.joint_state.position[self.camera_joint_indices['CAMERA_TILT']] = target_angle
            else:
                # In real mode, send command to camera
                self.send_real_hardware_command('camera_joint', target_angle)
            
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
            iteration_count = 0
            while time.time() - start_time < duration and self.trajectory_active:
                elapsed = time.time() - start_time
                t = min(1.0, elapsed / duration)
                t_smooth = 3 * t**2 - 2 * t**3
                
                # Calculate current positions
                current_positions = []
                for j in range(3):
                    pos = start_joints[j] + (target_joints[j] - start_joints[j]) * t_smooth
                    current_positions.append(pos)
                
                # In fake mode, update joint state for visualization
                # In real mode, only send commands - visualization comes from real robot
                if not self.use_real_hardware:
                    self.joint_state.position[self.arm_joint_indices['BEARING']] = current_positions[0]
                    self.joint_state.position[self.arm_joint_indices['SERVO_LOWER']] = current_positions[1]
                    self.joint_state.position[self.arm_joint_indices['SERVO_UPPER']] = current_positions[2]
                    
                    # Update GUI sliders (thread-safe) - only in fake mode
                    if hasattr(self, 'root') and self.root.winfo_exists():
                        self.root.after(0, self.update_gui_during_trajectory, current_positions)
                else:
                    # Send to real hardware at lower rate (every 5th iteration = 10Hz)
                    if iteration_count % 5 == 0:
                        self.send_arm_command_real(current_positions[0], current_positions[1], current_positions[2])
                
                iteration_count += 1
                time.sleep(0.02)  # 50Hz GUI update rate
            
            # Ensure final position is set
            if self.trajectory_active:
                if not self.use_real_hardware:
                    # Fake mode: update joint state and GUI
                    self.joint_state.position[self.arm_joint_indices['BEARING']] = target_joints[0]
                    self.joint_state.position[self.arm_joint_indices['SERVO_LOWER']] = target_joints[1]
                    self.joint_state.position[self.arm_joint_indices['SERVO_UPPER']] = target_joints[2]
                    
                    # Update GUI to final position
                    if hasattr(self, 'root') and self.root.winfo_exists():
                        self.root.after(0, self.update_gui_during_trajectory, target_joints)
                else:
                    # Real mode: only send final command
                    self.send_arm_command_real(target_joints[0], target_joints[1], target_joints[2])
                
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
            iteration_count = 0
            while time.time() - start_time < duration and self.trajectory_active:
                elapsed = time.time() - start_time
                t = min(1.0, elapsed / duration)
                t_smooth = 3 * t**2 - 2 * t**3
                
                # Calculate current positions
                current_positions = []
                for j in range(3):
                    pos = start_joints[j] + (target_joints[j] - start_joints[j]) * t_smooth
                    current_positions.append(pos)
                
                # In fake mode, update joint state and GUI
                # In real mode, only send commands - visualization comes from real robot
                if not self.use_real_hardware:
                    # Update joint state (arm only, camera not affected)
                    self.joint_state.position[self.arm_joint_indices['BEARING']] = current_positions[0]
                    self.joint_state.position[self.arm_joint_indices['SERVO_LOWER']] = current_positions[1]
                    self.joint_state.position[self.arm_joint_indices['SERVO_UPPER']] = current_positions[2]
                    
                    # Update GUI sliders (thread-safe) - only in fake mode
                    if hasattr(self, 'root') and self.root.winfo_exists():
                        self.root.after(0, self.update_gui_during_reset_trajectory, current_positions)
                else:
                    # Send to real hardware at lower rate (every 5th iteration = 10Hz)
                    if iteration_count % 5 == 0:
                        self.send_arm_command_real(current_positions[0], current_positions[1], current_positions[2])
                
                iteration_count += 1
                time.sleep(0.02)  # 50Hz update rate
            
            # Ensure final position is set
            if self.trajectory_active:
                if not self.use_real_hardware:
                    # Fake mode: update joint state and GUI
                    self.joint_state.position[self.arm_joint_indices['BEARING']] = target_joints[0]
                    self.joint_state.position[self.arm_joint_indices['SERVO_LOWER']] = target_joints[1]
                    self.joint_state.position[self.arm_joint_indices['SERVO_UPPER']] = target_joints[2]
                    
                    # Update GUI to final position
                    if hasattr(self, 'root') and self.root.winfo_exists():
                        self.root.after(0, self.update_gui_during_reset_trajectory, target_joints)
                else:
                    # Real mode: send final command
                    self.send_arm_command_real(target_joints[0], target_joints[1], target_joints[2])
                
                if hasattr(self, 'root') and self.root.winfo_exists():
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
            iteration_count = 0
            while time.time() - start_time < duration and self.trajectory_active:
                elapsed = time.time() - start_time
                t = min(1.0, elapsed / duration)
                
                # Smooth interpolation using cubic easing
                t_smooth = 3 * t**2 - 2 * t**3
                
                # Calculate current camera angle
                current_angle = start_angle + (target_angle - start_angle) * t_smooth
                
                # In fake mode, update joint state and GUI
                # In real mode, only send commands - visualization comes from real robot
                if not self.use_real_hardware:
                    # Update joint state
                    self.joint_state.position[self.camera_joint_indices['CAMERA_TILT']] = current_angle
                    
                    # Update GUI slider (thread-safe)
                    if hasattr(self, 'root') and self.root.winfo_exists():
                        self.root.after(0, self.update_gui_during_camera_trajectory, current_angle)
                else:
                    # Send to real hardware at lower rate (every 5th iteration = 10Hz)
                    if iteration_count % 5 == 0:
                        self.send_real_hardware_command('camera_joint', current_angle)
                
                iteration_count += 1
                time.sleep(0.02)  # 50Hz update rate
            
            # Ensure final position is set
            if self.trajectory_active:
                if not self.use_real_hardware:
                    # Fake mode: update joint state and GUI
                    self.joint_state.position[self.camera_joint_indices['CAMERA_TILT']] = target_angle
                    
                    # Update GUI to final position
                    if hasattr(self, 'root') and self.root.winfo_exists():
                        self.root.after(0, self.update_gui_during_camera_trajectory, target_angle)
                else:
                    # Real mode: send final command
                    self.send_real_hardware_command('camera_joint', target_angle)
                
                if hasattr(self, 'root') and self.root.winfo_exists():
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
            previous_above_threshold = False  # Track previous state to detect threshold crossings
            
            while time.time() - start_time < duration and self.trajectory_active:
                elapsed = time.time() - start_time
                t = min(1.0, elapsed / duration)
                t_smooth = 3 * t**2 - 2 * t**3
                
                # Check forces if closing
                if is_closing:
                    force_r2, force_l2 = self.get_gripper_forces()
                    
                    # Check if both forces exceed threshold
                    current_above_threshold = (force_r2 >= self.force_threshold and force_l2 >= self.force_threshold)
                    
                    # Detect threshold crossing (went from below to above threshold)
                    if current_above_threshold and not previous_above_threshold:
                        # Threshold just crossed - increment retry count
                        self.gripper_retry_count += 1
                        
                        # Get current positions at threshold crossing
                        current_positions = []
                        for j in range(4):
                            pos = start_joints[j] + (target_joints[j] - start_joints[j]) * t_smooth
                            current_positions.append(pos)
                        
                        # Update joint state to current position (hold at threshold position) - only in fake mode
                        if not self.use_real_hardware:
                            self.joint_state.position[self.gripper_joint_indices['L1']] = current_positions[0]
                            self.joint_state.position[self.gripper_joint_indices['L2']] = current_positions[1]
                            self.joint_state.position[self.gripper_joint_indices['R1']] = current_positions[2]
                            self.joint_state.position[self.gripper_joint_indices['R2']] = current_positions[3]
                        
                        # Notify GUI about threshold reached
                        if hasattr(self, 'root') and self.root.winfo_exists():
                            self.root.after(0, self.gripper_threshold_reached, current_positions, force_r2, force_l2, self.gripper_retry_count)
                        
                        # If we've reached max retries, stop completely
                        if self.gripper_retry_count > self.max_gripper_retries:
                            force_exceeded = True
                            self.trajectory_active = False
                            
                            # Notify GUI that gripper stopped after retries
                            if hasattr(self, 'root') and self.root.winfo_exists():
                                self.root.after(0, self.gripper_force_stopped, current_positions, force_r2, force_l2)
                            
                            break
                        
                        # Otherwise, wait 1 second before continuing (retry)
                        # Retry mechanism helps handle slip in simulation and ensures object is securely held
                        time.sleep(1.0)
                        
                        # Check if forces are still above threshold after delay
                        force_r2_check, force_l2_check = self.get_gripper_forces()
                        if force_r2_check >= self.force_threshold and force_l2_check >= self.force_threshold:
                            # Forces still above threshold - stop instead of continuing
                            force_exceeded = True
                            self.trajectory_active = False
                            
                            # Notify GUI that gripper stopped (threshold still met after delay)
                            if hasattr(self, 'root') and self.root.winfo_exists():
                                self.root.after(0, self.gripper_force_stopped, current_positions, force_r2_check, force_l2_check)
                            
                            break
                        
                        # Forces dropped below threshold - continue closing from current position
                        # Update start positions to current position for next retry attempt
                        start_joints = current_positions.copy()
                        start_time = time.time()  # Reset timer for next retry
                        previous_above_threshold = False  # Reset to detect next threshold crossing
                    
                    previous_above_threshold = current_above_threshold
                
                # Calculate current positions
                current_positions = []
                for j in range(4):
                    pos = start_joints[j] + (target_joints[j] - start_joints[j]) * t_smooth
                    current_positions.append(pos)
                
                # In fake mode, update joint state and GUI
                # In real mode, only send commands - visualization comes from real robot
                if not self.use_real_hardware:
                    # Update joint state
                    self.joint_state.position[self.gripper_joint_indices['L1']] = current_positions[0]
                    self.joint_state.position[self.gripper_joint_indices['L2']] = current_positions[1]
                    self.joint_state.position[self.gripper_joint_indices['R1']] = current_positions[2]
                    self.joint_state.position[self.gripper_joint_indices['R2']] = current_positions[3]
                    
                    # Update GUI sliders (thread-safe) - only in fake mode
                    if hasattr(self, 'root') and self.root.winfo_exists():
                        self.root.after(0, self.update_gripper_gui_during_trajectory, current_positions)
                else:
                    # Send to real hardware
                    # Use R1 position to calculate wrist angle
                    gripper_angle = abs(current_positions[2])  # R1 is index 2
                    wrist_angle = self.gripper_to_wrist_angle(gripper_angle)
                    self.send_gripper_command_real(wrist_angle)
                
                time.sleep(0.02)  # 50Hz update rate
            
            # Ensure final position is set
            if self.trajectory_active:
                if not self.use_real_hardware:
                    # Fake mode: update joint state and GUI
                    self.joint_state.position[self.gripper_joint_indices['L1']] = target_joints[0]
                    self.joint_state.position[self.gripper_joint_indices['L2']] = target_joints[1]
                    self.joint_state.position[self.gripper_joint_indices['R1']] = target_joints[2]
                    self.joint_state.position[self.gripper_joint_indices['R2']] = target_joints[3]
                    
                    # Update GUI to final position
                    if hasattr(self, 'root') and self.root.winfo_exists():
                        self.root.after(0, self.update_gripper_gui_during_trajectory, target_joints)
                else:
                    # Real mode: send final command
                    gripper_angle = abs(target_joints[2])  # R1 is index 2
                    wrist_angle = self.gripper_to_wrist_angle(gripper_angle)
                    self.send_gripper_command_real(wrist_angle)
                
                if hasattr(self, 'root') and self.root.winfo_exists():
                    self.root.after(0, self.gripper_trajectory_completed)
            
        except Exception as e:
            if hasattr(self, 'root') and self.root.winfo_exists():
                self.root.after(0, self.gripper_trajectory_error, str(e))
        finally:
            self.trajectory_active = False
            
    def update_gripper_gui_during_trajectory(self, joint_positions):
        """Update gripper GUI sliders during trajectory execution (called from main thread)
        Note: This should only be called in fake hardware mode"""
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
            
            # Only update joint state in fake hardware mode
            if not self.use_real_hardware:
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
        # Stop simulation clock monitoring
        self._stop_sim_clock_monitoring()
        
        self.gripper_status_text.insert(tk.END, f"Gripper trajectory completed successfully!\n\n")
        self.gripper_status_text.see(tk.END)
        self.update_status("Gripper trajectory completed")
    
    def gripper_threshold_reached(self, current_positions, force_r2, force_l2, retry_count):
        """Handle gripper threshold reached (called from main thread) - may continue or stop"""
        self.gripper_status_text.insert(tk.END, f"Force threshold reached (Attempt {retry_count}/{self.max_gripper_retries + 1}):\n")
        self.gripper_status_text.insert(tk.END, f"  R2 force: {force_r2:.3f} (threshold: {self.force_threshold})\n")
        self.gripper_status_text.insert(tk.END, f"  L2 force: {force_l2:.3f} (threshold: {self.force_threshold})\n")
        
        if retry_count <= self.max_gripper_retries:
            self.gripper_status_text.insert(tk.END, f"  Continuing to close (retry {retry_count}/{self.max_gripper_retries})...\n\n")
            self.update_status(f"Threshold reached - Retrying close ({retry_count}/{self.max_gripper_retries})")
        else:
            self.gripper_status_text.insert(tk.END, f"  Maximum retries reached - Stopping\n\n")
            self.update_status(f"Gripper stopped after {self.max_gripper_retries} retries")
        
        self.gripper_status_text.see(tk.END)
    
    def gripper_force_stopped(self, current_positions, force_r2, force_l2):
        """Handle gripper stopping due to force threshold after max retries (called from main thread)"""
        # Stop simulation clock monitoring
        self._stop_sim_clock_monitoring()
        
        self.gripper_status_text.insert(tk.END, f"Gripper stopped after {self.max_gripper_retries} retries!\n")
        self.gripper_status_text.insert(tk.END, f"  R2 force: {force_r2:.3f} (threshold: {self.force_threshold})\n")
        self.gripper_status_text.insert(tk.END, f"  L2 force: {force_l2:.3f} (threshold: {self.force_threshold})\n")
        self.gripper_status_text.insert(tk.END, f"  Final position: L1={current_positions[0]:.3f}, L2={current_positions[1]:.3f}, R1={current_positions[2]:.3f}, R2={current_positions[3]:.3f}\n\n")
        self.gripper_status_text.see(tk.END)
        self.update_status(f"Gripper stopped - Max retries reached (R2: {force_r2:.2f}, L2: {force_l2:.2f})")
        
    def gripper_trajectory_error(self, error_msg):
        """Handle gripper trajectory error (called from main thread)"""
        # Stop simulation clock monitoring
        self._stop_sim_clock_monitoring()
        
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
            import os
            
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
        # Only update internal state in fake hardware mode
        if not self.use_real_hardware:
            self.joint_state.position[self.gripper_joint_indices['L1']] = pos
        self.update_status(f"Left L1: {pos:.3f}")
        
    def on_left_l2_change(self, value):
        """Handle left L2 slider change"""
        pos = float(value)
        self.left_l2_label.config(text=f"{pos:.3f}")
        # Only update internal state in fake hardware mode
        if not self.use_real_hardware:
            self.joint_state.position[self.gripper_joint_indices['L2']] = pos
        self.update_status(f"Left L2: {pos:.3f}")
        
    def on_right_r1_change(self, value):
        """Handle right R1 slider change"""
        pos = float(value)
        self.right_r1_label.config(text=f"{pos:.3f}")
        # Only update internal state in fake hardware mode
        if not self.use_real_hardware:
            self.joint_state.position[self.gripper_joint_indices['R1']] = pos
        self.update_status(f"Right R1: {pos:.3f}")
        
    def on_right_r2_change(self, value):
        """Handle right R2 slider change"""
        pos = float(value)
        self.right_r2_label.config(text=f"{pos:.3f}")
        # Only update internal state in fake hardware mode
        if not self.use_real_hardware:
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
        right_r1_pos = joint_angle
        right_r2_pos = -joint_angle
        
        # Only update internal state in fake hardware mode
        if not self.use_real_hardware:
            self.joint_state.position[self.gripper_joint_indices['L1']] = left_pos
            self.joint_state.position[self.gripper_joint_indices['L2']] = left_pos
            self.joint_state.position[self.gripper_joint_indices['R1']] = right_r1_pos
            self.joint_state.position[self.gripper_joint_indices['R2']] = right_r2_pos
        else:
            # In real mode, only send command - visualization comes from real robot
            wrist_angle = self.gripper_to_wrist_angle(joint_angle)
            self.send_gripper_command_real(wrist_angle)
        
        # Update individual sliders if they exist (for display only)
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
            
            # Only update internal state in fake hardware mode
            if not self.use_real_hardware:
                self.joint_state.position[self.gripper_joint_indices['L1']] = -max_angle  # L1
                self.joint_state.position[self.gripper_joint_indices['L2']] = -max_angle  # L2
                self.joint_state.position[self.gripper_joint_indices['R1']] = max_angle   # R1 (direct mapping)
                self.joint_state.position[self.gripper_joint_indices['R2']] = -max_angle  # R2 (inverted mapping)
            else:
                # In real mode, send command to open gripper
                wrist_angle = self.gripper_to_wrist_angle(max_angle)
                self.send_gripper_command_real(wrist_angle)
            
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
            
            # Send to real hardware if in real mode
            if self.use_real_hardware:
                wrist_angle = self.gripper_to_wrist_angle(max_angle)
                self.send_gripper_command_real(wrist_angle)
        
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
                
                # Start simulation clock monitoring for gripper closing
                self._start_sim_clock_monitoring()
                
                # Reset retry counter for new close operation
                self.gripper_retry_count = 0
                self.force_threshold_reached = False
                
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
            
            # Only update internal state in fake hardware mode
            if not self.use_real_hardware:
                # Update joint positions
                self.joint_state.position[self.gripper_joint_indices['L1']] = 0.0  # L1
                self.joint_state.position[self.gripper_joint_indices['L2']] = 0.0  # L2
                self.joint_state.position[self.gripper_joint_indices['R1']] = 0.0  # R1
                self.joint_state.position[self.gripper_joint_indices['R2']] = 0.0  # R2
            else:
                # In real mode, send command to close gripper
                self.send_gripper_command_real(0.0)  # 0 = closed
            
            # Update gripper label
            if hasattr(self, 'gripper_label'):
                self.gripper_label.config(text="0.000 (Closed)")
            
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
        # Reset gripper joints to closed position (GUI sliders)
        self.left_l1_var.set(0.0)
        self.left_l2_var.set(0.0)
        self.right_r1_var.set(0.0)
        self.right_r2_var.set(0.0)
        
        # Update labels
        self.left_l1_label.config(text="0.000")
        self.left_l2_label.config(text="0.000")
        self.right_r1_label.config(text="0.000")
        self.right_r2_label.config(text="0.000")
        
        # Reset arm joints (GUI sliders)
        self.bearing_var.set(0.0)
        self.servo_lower_var.set(0.0)
        self.servo_upper_var.set(0.0)
        
        # Reset camera tilt (GUI slider)
        self.camera_tilt_var.set(0.0)
        
        # Only update internal state in fake hardware mode
        if not self.use_real_hardware:
            # Update gripper joint positions
            self.joint_state.position[self.gripper_joint_indices['L1']] = 0.0
            self.joint_state.position[self.gripper_joint_indices['L2']] = 0.0
            self.joint_state.position[self.gripper_joint_indices['R1']] = 0.0
            self.joint_state.position[self.gripper_joint_indices['R2']] = 0.0
            
            # Update arm joint positions
            self.joint_state.position[self.arm_joint_indices['BEARING']] = 0.0
            self.joint_state.position[self.arm_joint_indices['SERVO_LOWER']] = 0.0
            self.joint_state.position[self.arm_joint_indices['SERVO_UPPER']] = 0.0
            self.joint_state.position[self.camera_joint_indices['CAMERA_TILT']] = 0.0
        else:
            # In real mode, send commands to reset all joints
            self.send_real_hardware_command('base_joint', 0.0)
            self.send_real_hardware_command('shoulder_joint', 0.0)
            self.send_real_hardware_command('elbow_joint', 0.0)
            self.send_real_hardware_command('camera_joint', 0.0)
            self.send_gripper_command_real(0.0)  # Close gripper
        
        # Update labels
        self.bearing_label.config(text="0.000")
        self.servo_lower_label.config(text="0.000")
        self.servo_upper_label.config(text="0.000")
        self.camera_tilt_label.config(text="0.000 rad (0.0°)")
        
        self.update_status("Gripper, arm, and camera reset")
        
        # Send to real hardware if in real mode
        if self.use_real_hardware:
            self.send_arm_command_real(0.0, 0.0, 0.0)
            self.send_gripper_command_real(0.0)
            self.send_camera_command_real(0.0)
        
    def move_forward(self):
        """Send forward motion command"""
        # Use positive value for forward (intuitive: +5 for forward)
        forward_speed = 5.0
        forward_linear = 0.5
        
        # Apply wheel inversion compensation if needed
        if self.wheels_inverted:
            forward_speed = -forward_speed
            forward_linear = -forward_linear
        
        if self.use_real_hardware:
            # Hardware mode: use cmd_vel (Twist) for motor driver
            msg = Twist()
            msg.linear.x = forward_linear
            msg.angular.z = 0.0
            self.cmd_vel_pub.publish(msg)
            wheel_status = "inverted" if self.wheels_inverted else "normal"
            self.motion_status_text.insert(tk.END, f"Forward command sent (hardware): linear.x={forward_linear} (wheels: {wheel_status})\n")
        else:
            # Fake hardware mode: use Float64MultiArray for simulation
            # Pattern: [left_front, right_front, left_back, right_back]
            msg = Float64MultiArray()
            msg.data = [forward_speed, -forward_speed, forward_speed, -forward_speed]
            self.velocity_pub.publish(msg)
            wheel_status = "inverted" if self.wheels_inverted else "normal"
            self.motion_status_text.insert(tk.END, f"Forward command sent (simulation): {msg.data} (wheels: {wheel_status})\n")
        self.motion_status_text.see(tk.END)
        self.update_status("Moving forward")
        
    def move_backward(self):
        """Send backward motion command"""
        # Use negative value for backward (intuitive: -5 for backward)
        backward_speed = -5.0
        backward_linear = -0.5
        
        # Apply wheel inversion compensation if needed
        if self.wheels_inverted:
            backward_speed = -backward_speed
            backward_linear = -backward_linear
        
        if self.use_real_hardware:
            # Hardware mode: use cmd_vel (Twist) for motor driver
            msg = Twist()
            msg.linear.x = backward_linear
            msg.angular.z = 0.0
            self.cmd_vel_pub.publish(msg)
            wheel_status = "inverted" if self.wheels_inverted else "normal"
            self.motion_status_text.insert(tk.END, f"Backward command sent (hardware): linear.x={backward_linear} (wheels: {wheel_status})\n")
        else:
            # Fake hardware mode: use Float64MultiArray for simulation
            # Pattern: [left_front, right_front, left_back, right_back]
            msg = Float64MultiArray()
            msg.data = [backward_speed, -backward_speed, backward_speed, -backward_speed]
            self.velocity_pub.publish(msg)
            wheel_status = "inverted" if self.wheels_inverted else "normal"
            self.motion_status_text.insert(tk.END, f"Backward command sent (simulation): {msg.data} (wheels: {wheel_status})\n")
        self.motion_status_text.see(tk.END)
        self.update_status("Moving backward")
        
    def move_left(self):
        """Send left turn command"""
        # Use positive angular velocity for left turn (intuitive)
        angular_speed = 0.5
        turn_speed = 5.0
        
        # Apply wheel inversion compensation if needed
        if self.wheels_inverted:
            angular_speed = -angular_speed
            turn_speed = -turn_speed
        
        if self.use_real_hardware:
            # Hardware mode: use cmd_vel (Twist) for motor driver
            msg = Twist()
            msg.linear.x = 0.0
            msg.angular.z = angular_speed
            self.cmd_vel_pub.publish(msg)
            wheel_status = "inverted" if self.wheels_inverted else "normal"
            self.motion_status_text.insert(tk.END, f"Left turn command sent (hardware): angular.z={angular_speed} (wheels: {wheel_status})\n")
        else:
            # Fake hardware mode: use Float64MultiArray for simulation
            msg = Float64MultiArray()
            msg.data = [turn_speed, turn_speed, turn_speed, turn_speed]
            self.velocity_pub.publish(msg)
            wheel_status = "inverted" if self.wheels_inverted else "normal"
            self.motion_status_text.insert(tk.END, f"Left turn command sent (simulation): {msg.data} (wheels: {wheel_status})\n")
        self.motion_status_text.see(tk.END)
        self.update_status("Turning left")
        
    def move_right(self):
        """Send right turn command"""
        # Use negative angular velocity for right turn (intuitive)
        angular_speed = -0.5
        turn_speed = -5.0
        
        # Apply wheel inversion compensation if needed
        if self.wheels_inverted:
            angular_speed = -angular_speed
            turn_speed = -turn_speed
        
        if self.use_real_hardware:
            # Hardware mode: use cmd_vel (Twist) for motor driver
            msg = Twist()
            msg.linear.x = 0.0
            msg.angular.z = angular_speed
            self.cmd_vel_pub.publish(msg)
            wheel_status = "inverted" if self.wheels_inverted else "normal"
            self.motion_status_text.insert(tk.END, f"Right turn command sent (hardware): angular.z={angular_speed} (wheels: {wheel_status})\n")
        else:
            # Fake hardware mode: use Float64MultiArray for simulation
            msg = Float64MultiArray()
            msg.data = [turn_speed, turn_speed, turn_speed, turn_speed]
            self.velocity_pub.publish(msg)
            wheel_status = "inverted" if self.wheels_inverted else "normal"
            self.motion_status_text.insert(tk.END, f"Right turn command sent (simulation): {msg.data} (wheels: {wheel_status})\n")
        self.motion_status_text.see(tk.END)
        self.update_status("Turning right")
        
    def stop_motion(self):
        """Send stop motion command"""
        if self.use_real_hardware:
            # Hardware mode: use cmd_vel (Twist) for motor driver
            msg = Twist()
            msg.linear.x = 0.0
            msg.angular.z = 0.0
            self.cmd_vel_pub.publish(msg)
            self.motion_status_text.insert(tk.END, "Stop command sent (hardware): linear.x=0.0, angular.z=0.0\n")
        else:
            # Fake hardware mode: use Float64MultiArray for simulation
            msg = Float64MultiArray()
            msg.data = [0.0, 0.0, 0.0, 0.0]
            self.velocity_pub.publish(msg)
            self.motion_status_text.insert(tk.END, "Stop command sent (simulation): [0.0, 0.0, 0.0, 0.0]\n")
        self.motion_status_text.see(tk.END)
        self.update_status("Stopped")
        
    def _create_objects_subscription(self):
        """Create or recreate the objects subscription with current topic name"""
        # Destroy existing subscription if it exists
        if self.objects_sub is not None:
            self.destroy_subscription(self.objects_sub)
            self.objects_sub = None
        
        # Create new subscription with current topic
        self.objects_sub = self.create_subscription(
            TFMessage,
            self.objects_topic,
            self.objects_callback,
            10
        )
        self.get_logger().info(f'Subscribed to objects topic: {self.objects_topic}')
    
    def _update_objects_subscription(self):
        """Update the subscription to use the topic from GUI"""
        if hasattr(self, 'topic_var'):
            new_topic = self.topic_var.get().strip()
            if new_topic and new_topic != self.objects_topic:
                self.objects_topic = new_topic
                self._create_objects_subscription()
                self._update_button_text()
                # Log offset information for drop_poses topic
                if self.objects_topic == '/drop_poses':
                    if hasattr(self, 'error_text'):
                        self.error_text.insert(tk.END, f"Note: Z-offset of +0.05m (+50mm) will be applied for drop_poses topic\n")
                        self.error_text.see(tk.END)
                return True
        return False
    
    def _update_button_text(self):
        """Update the button text based on the current topic"""
        if hasattr(self, 'move_to_object_button'):
            if self.objects_topic == '/drop_poses':
                self.move_to_object_button.config(text="Move to Drop")
            else:
                self.move_to_object_button.config(text="Move to Grab")
    
    def objects_callback(self, msg):
        """Callback for objects_poses topic (TFMessage format)"""
        import math
        with self.objects_lock:
            # Store the latest objects data
            self.objects_data = {}
            for transform in msg.transforms:
                # Use child_frame_id as object name
                object_name = transform.child_frame_id
                
                # Extract position from transform
                pos = [
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z
                ]
                
                # Extract orientation quaternion
                quat = [
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                    transform.transform.rotation.w
                ]
                
                # Convert quaternion to roll, pitch, yaw (Euler angles)
                # Using standard conversion formula
                w, x, y, z = quat[3], quat[0], quat[1], quat[2]
                
                # Roll (x-axis rotation)
                sinr_cosp = 2 * (w * x + y * z)
                cosr_cosp = 1 - 2 * (x * x + y * y)
                roll = math.atan2(sinr_cosp, cosr_cosp)
                
                # Pitch (y-axis rotation)
                sinp = 2 * (w * y - z * x)
                if abs(sinp) >= 1:
                    pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
                else:
                    pitch = math.asin(sinp)
                
                # Yaw (z-axis rotation)
                siny_cosp = 2 * (w * z + x * y)
                cosy_cosp = 1 - 2 * (y * y + z * z)
                yaw = math.atan2(siny_cosp, cosy_cosp)
                
                self.objects_data[object_name] = {
                    'position': pos,
                    'orientation': quat,
                    'roll': roll,
                    'pitch': pitch,
                    'yaw': yaw,
                    'header': transform.header
                }
    
    def force_r2_callback(self, msg):
        """Callback for gripper R2 force topic"""
        with self.force_lock:
            self.gripper_force_r2 = abs(msg.data)  # Use absolute value for force
    
    def force_l2_callback(self, msg):
        """Callback for gripper L2 force topic"""
        with self.force_lock:
            self.gripper_force_l2 = abs(msg.data)  # Use absolute value for force
    
    def _start_sim_clock_monitoring(self):
        """Start monitoring simulation clock for gripper closing"""
        from builtin_interfaces.msg import Time
        
        with self.sim_clock_lock:
            # Subscribe to simulation clock if not already subscribed
            if self.sim_clock_sub is None:
                self.sim_clock_sub = self.create_subscription(
                    Clock,
                    '/clock_sim',
                    self.clock_callback,
                    10
                )
                self.get_logger().info('Started simulation clock monitoring for gripper closing')
            
            # Reset sim clock state
            self.sim_clock = None
            self.sim_time_offset = None
            self.use_sim_clock = True
            self.last_publish_sim_time = None
    
    def _stop_sim_clock_monitoring(self):
        """Stop monitoring simulation clock"""
        with self.sim_clock_lock:
            self.use_sim_clock = False
            # Note: We don't destroy the subscriber as it may be needed again
            # Just disable its usage
    
    def clock_callback(self, msg):
        """Callback for simulation clock - only active during gripper closing"""
        from builtin_interfaces.msg import Time
        
        if not self.use_sim_clock:
            return  # Only process if gripper closing is active
        
        with self.sim_clock_lock:
            # Store the clock time message
            clock_time = Time()
            clock_time.sec = msg.clock.sec
            clock_time.nanosec = msg.clock.nanosec
            self.sim_clock = clock_time
            
            # Set initial sim time offset on first clock message
            if self.sim_time_offset is None:
                self.sim_time_offset = msg.clock.sec + msg.clock.nanosec * 1e-9
                self.get_logger().info(f'Simulation clock offset set: {self.sim_time_offset}')
    
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
            
            # Apply z-offset for drop_poses topic
            z_position = position[2]
            if self.objects_topic == '/drop_poses':
                z_position = position[2] + 0.05  # Add 0.05m for drop poses
            
            # Convert from meters to millimeters
            x_mm = position[0] * 1000
            y_mm = position[1] * 1000
            z_mm = z_position * 1000
            
            # Update coordinate fields
            self.x_var.set(f"{x_mm:.1f}")
            self.y_var.set(f"{y_mm:.1f}")
            self.z_var.set(f"{z_mm:.1f}")
            
            # Use existing move_to_position function
            action_text = "dropping" if self.objects_topic == '/drop_poses' else "grabbing"
            self.error_text.insert(tk.END, f"Moving to {action_text} object '{object_name}' at position: X={x_mm:.1f}mm, Y={y_mm:.1f}mm, Z={z_mm:.1f}mm")
            if self.objects_topic == '/drop_poses':
                self.error_text.insert(tk.END, f" (original Z={position[2]*1000:.1f}mm, offset +50mm)\n")
            else:
                self.error_text.insert(tk.END, "\n")
            self.error_text.see(tk.END)
            self.update_status(f"Moving to {action_text} object '{object_name}'")
            
            # Call the existing move_to_position function
            self.move_to_position()
            
        except Exception as e:
            self.error_text.insert(tk.END, f"Error moving to object: {str(e)}\n")
            self.error_text.see(tk.END)
            self.update_status(f"Error: {str(e)}")
    
    def update_topic(self):
        """Update the subscription to use the topic from the GUI"""
        try:
            if self._update_objects_subscription():
                self.error_text.insert(tk.END, f"Updated subscription to topic: {self.objects_topic}\n")
                self.error_text.see(tk.END)
                self.update_status(f"Subscribed to: {self.objects_topic}")
            else:
                # Still update button text even if topic didn't change (in case it was set before GUI)
                self._update_button_text()
                # Log offset information if switching to drop_poses
                if self.objects_topic == '/drop_poses':
                    self.error_text.insert(tk.END, f"Already subscribed to: {self.objects_topic}\n")
                    self.error_text.insert(tk.END, f"Note: Z-offset of +0.05m (+50mm) will be applied for drop_poses topic\n")
                else:
                    self.error_text.insert(tk.END, f"Already subscribed to: {self.objects_topic}\n")
                self.error_text.see(tk.END)
        except Exception as e:
            self.error_text.insert(tk.END, f"Error updating topic: {str(e)}\n")
            self.error_text.see(tk.END)
            self.update_status(f"Error: {str(e)}")
    
    def refresh_objects(self):
        """Refresh and display available objects"""
        try:
            # Update subscription if topic changed
            self._update_objects_subscription()
            
            available_objects = self.list_available_objects()
            
            if available_objects:
                self.error_text.insert(tk.END, f"Available objects: {', '.join(available_objects)}\n")
                self.error_text.insert(tk.END, f"Topic: {self.objects_topic}\n")
                self.error_text.see(tk.END)
                self.update_status(f"Found {len(available_objects)} objects")
            else:
                self.error_text.insert(tk.END, f"No objects detected on topic: {self.objects_topic}\n")
                self.error_text.insert(tk.END, "Make sure the object detection system is running and topic is correct.\n")
                self.error_text.see(tk.END)
                self.update_status("No objects detected")
                
        except Exception as e:
            self.error_text.insert(tk.END, f"Error refreshing objects: {str(e)}\n")
            self.error_text.see(tk.END)
            self.update_status(f"Error: {str(e)}")
    
    def on_hardware_mode_change(self):
        """Handle hardware mode selection change"""
        mode = self.hardware_mode_var.get()
        use_real = (mode == "real")
        
        # Update hardware mode
        self.set_hardware_mode(use_real)
        
        # Update status indicator
        if hasattr(self, 'hardware_status_label'):
            if use_real:
                self.hardware_status_label.config(text="● Real", foreground="green")
            else:
                self.hardware_status_label.config(text="● Fake", foreground="blue")
        
        self.get_logger().info(f'Hardware mode changed to: {mode}')
    
    def update_status(self, message):
        """Update status label"""
        if hasattr(self, 'status_label'):
            self.status_label.config(text=message)
        
    def publish_joint_states(self):
        """Publish current joint states synchronized with simulation clock during gripper closing"""
        if self.running:
            # Use simulation clock only during gripper closing
            with self.sim_clock_lock:
                if self.use_sim_clock and self.sim_clock is not None:
                    # Use simulation clock timestamp
                    self.joint_state.header.stamp = self.sim_clock
                    
                    # Check if we should publish based on simulation time
                    current_sim_time = self.sim_clock.sec + self.sim_clock.nanosec * 1e-9
                    
                    if self.last_publish_sim_time is not None:
                        sim_delta = current_sim_time - self.last_publish_sim_time
                        min_period = 1.0 / self.publish_rate  # Minimum period for target rate
                        
                        # Only publish if enough simulation time has passed
                        if sim_delta < min_period:
                            return  # Skip this publish, too soon
                    
                    self.last_publish_sim_time = current_sim_time
                else:
                    # Use system clock when not closing gripper
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










