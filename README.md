# Hướng dẫn chạy dự án Drone

## Bước 1: Chạy mô hình drone và map

```bash
ros2 launch sjtu_drone_bringup sjtu_drone_bringup.launch.py
```

## Bước 2: Chạy thuật toán MPC CasADi
ros2 run drone_mpc_casadi drone_mpc_node --ros-args -r /cmd_vel:=/simple_drone/cmd_vel
```bash
ros2 run drone_mpc_casadi drone_mpc_node --ros-args -r /cmd_vel:=/simple_drone/cmd_vel
```

## Bước 3: Đặt mục tiêu cho drone
```bash
ros2 topic pub /goal geometry_msgs/msg/Point "{x: 2.0, y: 2.0, z: 3.0}"
```
