from setuptools import setup

package_name = 'drone_astar'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tuanhungjr',
    maintainer_email='your@email.com',
    description='A* path planning for drone in ROS2',
    license='MIT',
    entry_points={
        'console_scripts': [
            'drone_astar_node = drone_astar.drone_astar_node:main'
        ],
    },
)

