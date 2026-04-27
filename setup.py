import glob
from setuptools import setup

package_name = 'final_challenge'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, package_name + '.computer_vision'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/final_challenge/launch', glob.glob('launch/*.xml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rss2026-7',
    maintainer_email='rss2026-7@mit.edu',
    description='RSS 2026 Final Challenge — lane following and boating school',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lane_detector = final_challenge.lane_detector:main',
            'boundary_pure_pursuit = final_challenge.lane_follower:main',
            'test_lane_publisher = final_challenge.test_lane_publisher:main',
        ],
    },
)
