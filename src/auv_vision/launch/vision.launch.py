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
            executable='clahe_node',
            name='clahe_node',
            output='screen'
        )
    ])
