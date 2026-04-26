from setuptools import find_packages, setup

package_name = 'final_challenge2026'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Team 7',
    maintainer_email='todo@todo.todo',
    description='Mrs. Puff Boating School — RSS Final Challenge Part B',
    license='Apache License, Version 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'state_machine = final_challenge2026.state_machine:main',
            'basement_point_publisher = final_challenge2026.basement_point_publisher:main',
        ],
    },
)
