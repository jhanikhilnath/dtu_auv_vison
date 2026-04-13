from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='auv_vision',
            executable='camera_node',
            name='camera_node',
            output='screen'
        ),
        Node(
            package='auv_vision',
            executable='enhancement_node',
            name='enhancement_node',
            output='screen'
        )
    ])
