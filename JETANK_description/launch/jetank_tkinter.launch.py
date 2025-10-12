#!/usr/bin/env python3

from launch_ros.actions import Node
from launch import LaunchDescription
import xacro
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    share_dir = get_package_share_directory('JETANK_description')

    xacro_file = os.path.join(share_dir, 'urdf', 'JETANK.xacro')
    robot_description_config = xacro.process_file(xacro_file)
    robot_urdf = robot_description_config.toxml()

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
        jetank_control_gui_node,
        ee_pose_publisher_node,
        camera_pose_publisher_node,
        camera_to_ee_pose_publisher_node,
        rviz_node
    ])
