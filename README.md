#BƯỚC 1: chạy mô hình drone và map
ros2 launch sjtu_drone_bringup sjtu_drone_bringup.launch.py
#BƯỚC 2:chạy thuật toán mpc casadi
ros2 run drone_mpc_casadi drone_mpc_node --ros-args -r /cmd_vel:=/simple_drone/cmd_vel
#BƯỚC 3:set tọa độ 
ros2 topic pub /goal geometry_msgs/msg/Point "{x: 10.0, y: 15.0, z: 15.0}"
