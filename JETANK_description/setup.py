from setuptools import setup
import os
from glob import glob

package_name = 'JETANK_description'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        (os.path.join('share', package_name, 'meshes'), glob('meshes/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),
        (os.path.join('share', package_name, 'scripts'), glob('scripts/*.py'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='author',
    maintainer_email='todo@todo.com',
    description='The ' + package_name + ' package',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'jetank_control_gui = JETANK_description.jetank_control_gui:main',
            'ee_pose_publisher = JETANK_description.ee_pose_publisher:main',
            'camera_pose_publisher = JETANK_description.camera_pose_publisher:main',
            'camera_to_ee_pose_publisher = JETANK_description.camera_to_ee_pose_publisher:main',
            'verify_detections = JETANK_description.verify_detections:main',
            'randomize_legos = JETANK_description.randomize_legos:main',
            'joint_command_bridge = JETANK_description.joint_command_bridge:main',
            'cmd_vel_to_wheels = JETANK_description.cmd_vel_to_wheels:main',
            'world_tf_publisher = JETANK_description.world_tf_publisher:main',
        ],
    },
)
