#!/usr/bin/env python3
"""
Forward Kinematics Function for JETANK Robot
Usage: python3 fk.py theta0 theta1 theta3
"""

import numpy as np
import math
import sys

def forward_kinematics(theta0, theta1, theta3):
    """
    Calculate forward kinematics for JETANK robot
    
    Args:
        theta0: Joint 0 angle (revolute_BEARING) in radians
        theta1: Joint 1 angle (Revolute_SERVO_LOWER) in radians  
        theta3: Joint 3 angle (Revolute_SERVO_UPPER) in radians
    
    Returns:
        tuple: (x, y, z) position in mm, or None if error
    """
    
    # Joint limits (from URDF)
    limits = [
        {'lower': -1.5708, 'upper': 1.5708},      # Joint 0
        {'lower': 0.0,     'upper': 1.570796},    # Joint 1
        {'lower': -3.1418, 'upper': 0.785594}     # Joint 3
    ]
    
    # Check joint limits
    if theta0 < limits[0]['lower'] or theta0 > limits[0]['upper']:
        print(f"Error: Joint 0 angle {theta0:.4f} outside limits [{limits[0]['lower']:.4f}, {limits[0]['upper']:.4f}]")
        return None
    if theta1 < limits[1]['lower'] or theta1 > limits[1]['upper']:
        print(f"Error: Joint 1 angle {theta1:.4f} outside limits [{limits[1]['lower']:.4f}, {limits[1]['upper']:.4f}]")
        return None
    if theta3 < limits[2]['lower'] or theta3 > limits[2]['upper']:
        print(f"Error: Joint 3 angle {theta3:.4f} outside limits [{limits[2]['lower']:.4f}, {limits[2]['upper']:.4f}]")
        return None
    
    # DH parameters (in meters)
    dh_params = [
        {'alpha': -1.57, 'a': -13.50/1000, 'd': 75.00/1000},  # Link 0
        {'alpha': 1.57,  'a': 0.00/1000,   'd': 0.00/1000},   # Link 1  
        {'alpha': -1.57, 'a': 0.00/1000,   'd': 95.00/1000},  # Link 2
        {'alpha': 1.57,  'a': -179.25/1000, 'd': 0.00/1000}   # Link 3
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
    
    return (position[0], position[1], position[2])

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 fk.py theta0 theta1 theta3")
        print("Example: python3 fk.py 0.0 1.57 0.0")
        print("\nJoint Limits:")
        print("Joint 0 (revolute_BEARING):     -1.5708 to 1.5708 rad")
        print("Joint 1 (Revolute_SERVO_LOWER): 0.0 to 1.570796 rad")
        print("Joint 3 (Revolute_SERVO_UPPER): -3.1418 to 0.785594 rad")
        sys.exit(1)
    
    try:
        theta0 = float(sys.argv[1])
        theta1 = float(sys.argv[2])
        theta3 = float(sys.argv[3])
        
        result = forward_kinematics(theta0, theta1, theta3)
        
        if result is not None:
            x, y, z = result
            print(f"Gripper center position:")
            print(f"x: {x:.2f} mm")
            print(f"y: {y:.2f} mm")
            print(f"z: {z:.2f} mm")
        else:
            sys.exit(1)
            
    except ValueError:
        print("Error: Please enter valid numbers")
        sys.exit(1)

if __name__ == "__main__":
    main()
