from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([

        Node(
            package='tactile_sensor',
            executable='tactile_node',
            name='tactile_node',
            parameters=[{
                'port': '/dev/ttyUSB0',
                'baud': 2_000_000,
                'rows': 12,
                'cols': 32,
                'thresh': 20,
                'noise_scale': 60,
                'init_frames': 30,
                'alpha': 0.2,
            }]
        ),

        # Tu décommenteras quand tu ajouteras la caméra
        # Node(
        #     package='camera',
        #     executable='camera_node',
        #     name='camera_node',
        #     parameters=[{'device': '/dev/video0'}]
        # ),

    ])