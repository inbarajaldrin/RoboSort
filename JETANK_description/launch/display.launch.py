from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
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

    gui_arg = DeclareLaunchArgument(
        name='gui',
        default_value='True'
    )

    show_gui = LaunchConfiguration('gui')

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {'robot_description': robot_urdf}
        ]
    )

    joint_state_publisher_node = Node(
        condition=UnlessCondition(show_gui),
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher'
    )

    joint_state_publisher_gui_node = Node(
        condition=IfCondition(show_gui),
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui'
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        output='screen'
    )

    return LaunchDescription([
        gui_arg,
        robot_state_publisher_node,
        joint_state_publisher_node,
        joint_state_publisher_gui_node,
        rviz_node
    ])
