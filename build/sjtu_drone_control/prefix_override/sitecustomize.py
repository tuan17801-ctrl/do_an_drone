import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/tuanhungjr/ros2_ws/src/install/sjtu_drone_control'
