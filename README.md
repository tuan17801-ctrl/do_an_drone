# Hướng dẫn chạy dự án Drone

## Bước 1: Chạy mô hình drone và map
```bash
ros2 launch sjtu_drone_bringup sjtu_drone_bringup.launch.py
## Bước 2:Chạy thuật toán MPC CasADi
ros2 run drone_mpc_casadi drone_mpc_node --ros-args -r /cmd_vel:=/simple_drone/cmd_vel

