#!/usr/bin/env python3

from launch_ros.actions import Node
from launch import LaunchDescription
import xacro
import os
import tempfile
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    share_dir = get_package_share_directory('JETANK_description')

    xacro_file = os.path.join(share_dir, 'urdf', 'JETANK.xacro')
    # Read and preprocess xacro file to replace $(find ...) with actual paths
    with open(xacro_file, 'r') as f:
        xacro_content = f.read()
    
    # Replace $(find JETANK_description) with actual package path
    xacro_content = xacro_content.replace('$(find JETANK_description)', share_dir)
    
    # Write preprocessed content to temporary file and process it
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xacro', delete=False) as tmp_file:
        tmp_file.write(xacro_content)
        tmp_xacro_file = tmp_file.name
    
    try:
        robot_description_config = xacro.process_file(tmp_xacro_file)
        robot_urdf = robot_description_config.toxml()
    finally:
        # Clean up temporary file
        os.unlink(tmp_xacro_file)

    rviz_config_file = os.path.join(share_dir, 'config', 'display.rviz')

    # Robot state publisher node
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {'robot_description': robot_urdf}
        ]
    )

    # Servo Driver - controls the 5 servos on the real robot
    # Remap joint_states to real_joint_states so GUI can subscribe
    servo_driver_node = Node(
        package='jetank_control',
        executable='servo',
        name='servo_driver',
        output='screen',
        remappings=[
            ('joint_states', 'real_joint_states'),  # Servo publishes real state here
        ]
    )

    # Motor Driver - controls the drive motors in hardware mode
    motor_driver_node = Node(
        package='jetank_control',
        executable='motor',
        name='motor_driver',
        output='screen'
    )

    # JETANK Control GUI
    jetank_control_gui_node = Node(
        package='JETANK_description',
        executable='jetank_control_gui',
        name='jetank_control_gui',
        output='screen'
    )

    # RViz node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        output='screen'
    )

    # End-Effector Pose Publisher
    ee_pose_publisher_node = Node(
        package='JETANK_description',
        executable='ee_pose_publisher',
        name='ee_pose_publisher',
        parameters=[
            {'base_frame': 'BEARING_1'},
            {'ee_frame': 'GRIPPER_CENTER_LINK'},
            {'publish_rate': 10.0},
            {'startup_delay': 3.0}  # Wait 3 seconds for TF tree to be ready
        ],
        output='screen'
    )

    # Camera Pose Publisher
    camera_pose_publisher_node = Node(
        package='JETANK_description',
        executable='camera_pose_publisher',
        name='camera_pose_publisher',
        parameters=[
            {'base_frame': 'BEARING_1'},
            {'camera_frame': 'CAMERA_1'},
            {'publish_rate': 10.0},
            {'startup_delay': 3.0}  # Wait 3 seconds for TF tree to be ready
        ],
        output='screen'
    )

    # Camera to End-Effector Pose Publisher
    camera_to_ee_pose_publisher_node = Node(
        package='JETANK_description',
        executable='camera_to_ee_pose_publisher',
        name='camera_to_ee_pose_publisher',
        parameters=[
            {'camera_frame': 'CAMERA_1'},
            {'ee_frame': 'GRIPPER_CENTER_LINK'},
            {'publish_rate': 10.0},
            {'startup_delay': 3.0}  # Wait 3 seconds for TF tree to be ready
        ],
        output='screen'
    )

    return LaunchDescription([
        robot_state_publisher_node,
        servo_driver_node,
        motor_driver_node,
        jetank_control_gui_node,
        ee_pose_publisher_node,
        camera_pose_publisher_node,
        camera_to_ee_pose_publisher_node,
        rviz_node
    ])


