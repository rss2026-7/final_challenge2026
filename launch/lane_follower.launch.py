"""Launch file for the BoundaryPurePursuit lane follower node."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('final_challenge')

    default_params_file = os.path.join(
        pkg_share, 'config', 'lane_follower_params.yaml'
    )

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params_file,
        description='Path to the lane follower parameter YAML file',
    )

    lane_follower_node = Node(
        package='final_challenge',
        executable='lane_follower',
        name='boundary_pure_pursuit',
        parameters=[LaunchConfiguration('params_file')],
        output='screen',
    )

    return LaunchDescription([
        params_file_arg,
        lane_follower_node,
    ])
