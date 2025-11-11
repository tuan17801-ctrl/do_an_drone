from setuptools import find_packages, setup

package_name = 'drone_mpc_casadi'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tuanhungjr',
    maintainer_email='tuan17801@gmail.com',
    description='Drone MPC controller using CasADi in ROS2 + Gazebo',
    license='MIT',
    tests_require=['pytest'],
   entry_points={
    'console_scripts': [
        'drone_mpc_node = drone_mpc_casadi.mpc_node:main',
    ],
},

)
