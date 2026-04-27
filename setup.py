import glob
import os
from setuptools import find_packages, setup

package_name = 'final_challenge'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        # Ament index marker
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        # Package manifest
        ('share/' + package_name, ['package.xml']),
        # Launch files
        ('share/' + package_name + '/launch',
         glob.glob(os.path.join('launch', '*launch.*'))),
        # Config / parameter files
        ('share/' + package_name + '/config',
         glob.glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kevinhuang',
    maintainer_email='kevinhuang@todo.todo',
    description='RSS Final Challenge 2026 – lane-following controller',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lane_follower = final_challenge.lane_follower:main',
            'test_lane_publisher = final_challenge.test_lane_publisher:main',
            'state_machine = final_challenge2026.state_machine:main',
            'basement_point_publisher = final_challenge2026.basement_point_publisher:main',

        ],
    },
)
