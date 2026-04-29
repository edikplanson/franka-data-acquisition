from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    return LaunchDescription([

        Node(
            package='tactile_pkg',
            executable='tactile_node'
        ),

        Node(
            package='camera_pkg',
            executable='camera_node'
        ),

        Node(
            package='robot_state_pkg',
            executable='state_node'
        ),

        Node(
            package='dataset_logger',
            executable='dataset_logger'
        )
    ])