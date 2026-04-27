"""
Combined launch file for testing BoundaryPurePursuit in racecar_simulator.

Brings up:
  1. nav2 map_server + lifecycle_manager
  2. racecar_model (URDF / robot_state_publisher)
  3. racecar_simulator  (2-D Ackermann sim)
  4. lane_follower       (BoundaryPurePursuit controller)
  5. test_lane_publisher  (synthetic lane-line feeder)

Usage (inside the Docker container):
  ros2 launch final_challenge lane_follower_sim_test.launch.py
  ros2 launch final_challenge lane_follower_sim_test.launch.py mode:=curve_left track_side:=left
  ros2 launch final_challenge lane_follower_sim_test.launch.py mode:=dropout map_name:=building_31
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    sim_share = get_package_share_directory('racecar_simulator')
    fc_share = get_package_share_directory('final_challenge')

    # ── Launch arguments ──────────────────────────────────────────────
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='straight',
        description='test_lane_publisher mode: straight | curve_left | curve_right | dropout',
    )
    map_name_arg = DeclareLaunchArgument(
        'map_name',
        default_value='stata_basement',
        description='Map base name (without extension) from racecar_simulator/maps/',
    )
    track_side_arg = DeclareLaunchArgument(
        'track_side',
        default_value='left',
        description='Which lane boundary to follow: left | right',
    )
    lane_width_arg = DeclareLaunchArgument(
        'lane_width',
        default_value='0.6',
        description='Simulated lane width in meters for test_lane_publisher',
    )
    curve_radius_arg = DeclareLaunchArgument(
        'curve_radius',
        default_value='3.0',
        description='Curve radius in meters for test_lane_publisher curve modes',
    )

    # ── Map server ────────────────────────────────────────────────────
    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{
            'yaml_filename': [
                sim_share, '/maps/', LaunchConfiguration('map_name'), '.yaml'
            ],
        }],
    )

    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager',
        output='screen',
        parameters=[{
            'autostart': True,
            'node_names': ['map_server'],
        }],
    )

    # ── Racecar model (URDF) ─────────────────────────────────────────
    racecar_model = IncludeLaunchDescription(
        XMLLaunchDescriptionSource(
            os.path.join(sim_share, 'launch', 'racecar_model.launch.xml')
        ),
    )

    # ── Racecar simulator ─────────────────────────────────────────────
    sim_params = os.path.join(sim_share, 'params.yaml')

    racecar_simulator = Node(
        package='racecar_simulator',
        executable='simulate',
        name='racecar_simulator',
        output='screen',
        parameters=[sim_params],
    )

    # ── Lane follower (node under test) ───────────────────────────────
    lane_follower_params = os.path.join(fc_share, 'config', 'lane_follower_params.yaml')

    lane_follower = Node(
        package='final_challenge',
        executable='lane_follower',
        name='boundary_pure_pursuit',
        output='screen',
        parameters=[
            lane_follower_params,
            {'track_side': LaunchConfiguration('track_side')},
        ],
    )

    # ── Synthetic lane-line publisher ─────────────────────────────────
    test_lane_publisher = Node(
        package='final_challenge',
        executable='test_lane_publisher',
        name='test_lane_publisher',
        output='screen',
        parameters=[{
            'mode': LaunchConfiguration('mode'),
            'lane_width': LaunchConfiguration('lane_width'),
            'curve_radius': LaunchConfiguration('curve_radius'),
        }],
    )

    # ── Assemble ──────────────────────────────────────────────────────
    return LaunchDescription([
        mode_arg,
        map_name_arg,
        track_side_arg,
        lane_width_arg,
        curve_radius_arg,
        map_server,
        lifecycle_manager,
        racecar_model,
        racecar_simulator,
        lane_follower,
        test_lane_publisher,
    ])
