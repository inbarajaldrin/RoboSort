#!/usr/bin/env python3
# Reference: ~/Projects/Exploring-VLAs/vla_SO-ARM101/src/so_arm101_control/launch/gazebo.launch.py
"""
Launch JETANK in Gazebo Harmonic with camera sensors and colored objects.
Spawns the JETANK URDF into Gazebo via ros_gz_sim, bridges camera topics to ROS2.

macOS porting: server/GUI split, GZ_ env vars, gz.msgs bridge types.
"""

import launch
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import xacro
import os
import atexit
import subprocess
import tempfile
from ament_index_python.packages import get_package_share_directory


def _kill_gz_sim():
    """Kill orphan gz sim server/gui on exit.

    These processes create their own process groups (setsid), so SIGINT
    from the ROS2 launch system never reaches them.  Python's atexit
    fires reliably when the interpreter shuts down after SIGINT.
    """
    subprocess.run(['pkill', '-f', 'gz sim'], capture_output=True)


atexit.register(_kill_gz_sim)


def generate_launch_description():
    headless_arg = DeclareLaunchArgument(
        'headless', default_value='false',
        description='Run Gazebo in headless mode (server only, no GUI window)')

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

    # Resource path so Gazebo can find package:// meshes
    install_share_parent = os.path.dirname(share_dir)

    # Also include drop_container_description meshes if available
    try:
        drop_share = get_package_share_directory('drop_container_description')
        drop_share_parent = os.path.dirname(drop_share)
    except Exception:
        drop_share_parent = ''

    resource_path = ':'.join(filter(None, [
        install_share_parent,
        drop_share_parent,
        os.environ.get('GZ_SIM_RESOURCE_PATH', ''),
    ]))

    # Plugin path so Gazebo finds system plugins in the conda env
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    plugin_path = ':'.join(filter(None, [
        os.path.join(conda_prefix, 'lib') if conda_prefix else '',
        os.environ.get('GZ_SIM_SYSTEM_PLUGIN_PATH', ''),
    ]))

    # --- Gazebo sim ---
    # macOS: must split server (-s) and GUI (-g) into separate processes
    headless = LaunchConfiguration('headless')
    gz_server_args = '-r -s ' + world_file

    gz_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py',
            ])
        ]),
        launch_arguments={'gz_args': gz_server_args}.items(),
    )

    # Separate GUI client (macOS requirement: can't combine server+GUI)
    gz_gui = ExecuteProcess(
        cmd=['gz', 'sim', '-g'],
        condition=UnlessCondition(headless),
        additional_env={
            'GZ_SIM_RESOURCE_PATH': resource_path,
        },
        output='screen',
    )

    # --- Robot State Publisher (publishes TF tree from URDF) ---
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'robot_description': robot_urdf,
            'use_sim_time': True,
        }],
    )

    # --- Spawn JETANK into Gazebo from robot_description topic ---
    gz_spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'jetank',
            '-allow_renaming', 'true',
            '-Y', '3.14159',  # Rotate 180° so camera faces the legos (+X)
        ],
    )

    # --- JETANK Control GUI (publishes /joint_states at 10Hz, accepts /joint_commands) ---
    jetank_control_gui = Node(
        package='JETANK_description',
        executable='jetank_control_gui',
        name='jetank_control_gui',
        output='screen',
    )

    # --- ros_gz_bridge: camera/clock (Gz→ROS) + joint commands (ROS→Gz) ---
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            # Camera topics: Gz → ROS
            '/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '/depth_camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            # Joint command topics: ROS → Gz
            '/cmd/revolute_BEARING@std_msgs/msg/Float64]gz.msgs.Double',
            '/cmd/Revolute_SERVO_LOWER@std_msgs/msg/Float64]gz.msgs.Double',
            '/cmd/Revolute_SERVO_UPPER@std_msgs/msg/Float64]gz.msgs.Double',
            '/cmd/revolute_CAMERA_HOLDER_ARM_LOWER@std_msgs/msg/Float64]gz.msgs.Double',
            '/cmd/revolute_GRIPPER_L1@std_msgs/msg/Float64]gz.msgs.Double',
            '/cmd/revolute_GRIPPER_L2@std_msgs/msg/Float64]gz.msgs.Double',
            '/cmd/Revolute_GRIPPER_R1@std_msgs/msg/Float64]gz.msgs.Double',
            '/cmd/Revolute_GRIPPER_R2@std_msgs/msg/Float64]gz.msgs.Double',
            # Free wheels: no controller — they roll freely with ground friction
            # DiffDrive: cmd_vel ROS→Gz (replaces per-wheel velocity controllers)
            '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
        ],
        output='screen',
    )

    # --- Joint command bridge: splits /joint_commands into per-joint Float64 topics ---
    joint_command_bridge = Node(
        package='JETANK_description',
        executable='joint_command_bridge',
        name='joint_command_bridge',
        output='screen',
    )

    # cmd_vel_to_wheels: no longer needed — DiffDrive plugin handles cmd_vel → wheels

    # --- EE pose publisher (delayed to let TF tree establish) ---
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

    # --- Camera pose publisher (delayed to let TF tree establish) ---
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

    # --- Camera to EE pose publisher (delayed to let TF tree establish) ---
    camera_to_ee_pose_publisher = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='JETANK_description',
                executable='camera_to_ee_pose_publisher',
                name='camera_to_ee_pose_publisher',
                output='screen',
            )
        ]
    )

    return LaunchDescription([
        headless_arg,

        # 0. Set resource paths so Gazebo can find package:// meshes and plugins
        SetEnvironmentVariable(
            name='GZ_SIM_RESOURCE_PATH',
            value=resource_path),
        SetEnvironmentVariable(
            name='IGN_GAZEBO_RESOURCE_PATH',
            value=resource_path),
        SetEnvironmentVariable(
            name='GZ_SIM_SYSTEM_PLUGIN_PATH',
            value=plugin_path),

        # 1. Start Gazebo server (always with -s on macOS)
        gz_server,
        # 1b. Start Gazebo GUI separately (macOS requirement)
        gz_gui,

        # 2. Publish robot description
        robot_state_publisher,

        # 3. Spawn JETANK into Gazebo
        gz_spawn_entity,

        # 4. After spawn completes, start GUI + bridge + pose publishers
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=gz_spawn_entity,
                on_exit=[
                    jetank_control_gui,
                    bridge,
                    joint_command_bridge,
                    ee_pose_publisher,
                    camera_pose_publisher,
                    camera_to_ee_pose_publisher,
                    # World TF: publishes world → base_footprint from Gazebo model pose
                    Node(
                        package='JETANK_description',
                        executable='world_tf_publisher',
                        name='world_tf_publisher',
                        output='screen',
                    ),
                ],
            )
        ),
    ])
