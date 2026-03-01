#!/usr/bin/env python3
"""
Launch JETANK in Ignition Gazebo Fortress with camera sensors and colored objects.
Uses ros_gz_bridge to bridge Ignition topics to ROS2.
"""

from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, SetEnvironmentVariable
from launch_ros.actions import Node
import xacro
import os
import tempfile
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    share_dir = get_package_share_directory('JETANK_description')

    # Process xacro to get robot URDF
    xacro_file = os.path.join(share_dir, 'urdf', 'JETANK.xacro')
    with open(xacro_file, 'r') as f:
        xacro_content = f.read()
    xacro_content = xacro_content.replace('$(find JETANK_description)', share_dir)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.xacro', delete=False) as tmp_file:
        tmp_file.write(xacro_content)
        tmp_xacro_file = tmp_file.name

    try:
        robot_description_config = xacro.process_file(tmp_xacro_file)
        robot_urdf = robot_description_config.toxml()
    finally:
        os.unlink(tmp_xacro_file)

    world_file = os.path.join(share_dir, 'worlds', 'lego_world.sdf')

    # Robot state publisher (publishes TF tree from URDF)
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{'robot_description': robot_urdf}],
    )

    # Joint state publisher (non-GUI, publishes default joint states)
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
    )

    # Set DISPLAY for Ignition Gazebo GUI
    set_display = SetEnvironmentVariable('DISPLAY', ':1')

    # Launch Ignition Gazebo server-only (headless) for faster performance
    # Use 'ign gazebo -r' (without -s) if you need the GUI
    ign_gazebo = ExecuteProcess(
        cmd=['ign', 'gazebo', '-r', '-s', world_file],
        output='screen',
        additional_env={'DISPLAY': ':1'},
    )

    # Robot is embedded directly in the world SDF (no dynamic spawning needed)

    # ros_gz_bridge: bridge camera/depth/clock topics from Ignition to ROS2
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/camera/image_raw@sensor_msgs/msg/Image@ignition.msgs.Image',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo@ignition.msgs.CameraInfo',
            '/depth_camera/image_raw@sensor_msgs/msg/Image@ignition.msgs.Image',
            '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
        ],
        output='screen',
    )

    # EE pose publisher (delayed to let TF tree establish)
    ee_pose_publisher = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='JETANK_description',
                executable='ee_pose_publisher',
                name='ee_pose_publisher',
                output='screen',
            )
        ]
    )

    # Camera pose publisher (delayed to let TF tree establish)
    camera_pose_publisher = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='JETANK_description',
                executable='camera_pose_publisher',
                name='camera_pose_publisher',
                output='screen',
            )
        ]
    )

    return LaunchDescription([
        set_display,
        robot_state_publisher,
        joint_state_publisher,
        ign_gazebo,
        bridge,
        ee_pose_publisher,
        camera_pose_publisher,
    ])
