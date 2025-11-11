#!/usr/bin/env python3 
# mpc_node_fixed_with_obstacles_logging.py
"""
ROS2 MPC Node for drone with obstacle avoidance (ready-to-run)

Enhancements:
- Logs minimum distance to obstacles for each control step.
- Saves CSV file 'mpc_log.csv' for offline analysis.
- Publishes predicted trajectory for RViz / debugging.
"""

import threading
import time
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import TwistStamped, Point
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float32MultiArray
import csv

try:
    import casadi as ca
except Exception:
    raise RuntimeError("CasADi not installed. Install with: pip install casadi")


class MPCNode(Node):
    def __init__(self):
        super().__init__('mpc_node_fixed')

        # ---------- Parameters ----------
        self.dt = 0.15        # MPC prediction timestep
        self.N = 12
        self.pub_hz = 20.0
        self.safe_radius = 4.0
        self.max_obs = 10
        self.mass = 1.477
        self.max_force = 25.0
        self.max_acc = self.max_force / self.mass
        self.Qp = 10.0
        self.Qv = 0.1
        self.R = 0.05
        self.frame = 'enu'
        self.ipopt_time_limit = 0.1

        # ---------- States ----------
        self.goal = np.array([5.0, 0.0, 1.0], dtype=float)
        self.current_pos = np.zeros(3, dtype=float)
        self.current_vel = np.zeros(3, dtype=float)
        self.current_yaw = 0.0
        self.obstacles = []        # dynamic obstacles from gazebo
        self.static_obstacles = [] # obstacles loaded from world
        self.odom_received = False

        # ---------- Internal ----------
        self.x_prev = None
        self.latest_cmd = TwistStamped()
        self.latest_cmd.header.frame_id = "base_link"
        self._stop_event = threading.Event()

        # CSV logging
        self.csv_file = open("mpc_log.csv", "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["time_sec","pos_x","pos_y","pos_z",
                                  "goal_x","goal_y","goal_z","min_dist_obs"])

        # Preload static obstacles
        self.load_static_obstacles()

        qos = QoSProfile(depth=10)
        self.cmd_pub = self.create_publisher(TwistStamped, '/simple_drone/cmd_vel', qos)
        self.traj_pub = self.create_publisher(Float32MultiArray, '/mpc/predicted_traj', qos)

        # subscriptions
        self.create_subscription(Odometry, '/simple_drone/odom', self.odom_cb, 10)
        self.create_subscription(ModelStates, '/gazebo/model_states', self.model_states_cb, 10)
        self.create_subscription(Point, '/goal', self.goal_cb, 10)

        # Build MPC
        self.build_mpc()

        # Timers & threads
        self.pub_timer = self.create_timer(1.0 / self.pub_hz, self.pub_loop)
        self.mpc_thread = threading.Thread(target=self.mpc_worker, daemon=True)
        self.mpc_thread.start()

        self.get_logger().info(f"[OK] MPC node started at {self.pub_hz} Hz (horizon={self.N}, dt={self.dt})")

    # ------------------------------
    # Load static obstacles
    # ------------------------------
    def load_static_obstacles(self):
        static = [
            ("House 1", 16.1928, 6.59969, 0.0, 3.0),
            ("House 2", -7.88791, 10.0389, 0.0, 3.0),
            ("House 3", -12.0321, -0.662221, 0.0, 3.0),
            ("apartment", 26.0781, -20.6711, 0.0, 4.0),
            ("oak_tree", 2.33516, -7.36935, 0.0, 1.5),
            ("pine_tree", -9.77152, -11.6186, 0.0, 1.5),
            ("post_office", -20.7018, -17.3801, 0.0, 3.0),
        ]
        self.static_obstacles = [[x, y, z, r] for (_, x, y, z, r) in static]
        self.get_logger().info(f"[OK] Loaded {len(self.static_obstacles)} static obstacles")

    # ------------------------------
    # Callbacks
    # ------------------------------
    def goal_cb(self, msg):
        self.goal = np.array([msg.x, msg.y, msg.z], dtype=float)
        self.get_logger().info(f"[Goal] New target: {self.goal}")

    def odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.twist.linear
        self.current_pos = np.array([p.x, p.y, p.z], dtype=float)
        self.current_vel = np.array([v.x, v.y, v.z], dtype=float)
        self.current_yaw = self.quat_to_yaw(q.w, q.x, q.y, q.z)
        self.odom_received = True

    def model_states_cb(self, msg):
        obs = []
        for i, name in enumerate(msg.name):
            lower = name.lower()
            if any(k in lower for k in ["tree", "house", "apartment", "post", "office", "pine", "oak", "box", "wall"]):
                p = msg.pose[i].position
                if "house" in lower or "apartment" in lower or "post" in lower:
                    r = 3.0
                elif "tree" in lower:
                    r = 1.5
                else:
                    r = 1.0
                obs.append([p.x, p.y, p.z, r])
        merged = list(self.static_obstacles)
        for o in obs:
            if any(np.linalg.norm(np.array(o[:2]) - np.array(s[:2])) < 0.5 for s in self.static_obstacles):
                continue
            merged.append(o)
        self.obstacles = merged[:self.max_obs]

    # ------------------------------
    # Build MPC
    # ------------------------------
    def build_mpc(self):
        nx, nu, N, dt, max_obs = 6, 3, self.N, self.dt, self.max_obs
        x = ca.SX.sym('x', nx)
        u = ca.SX.sym('u', nu)
        f = ca.Function('f', [x, u], [ca.vertcat(x[3:6], u)])

        X = ca.SX.sym('X', nx, N + 1)
        U = ca.SX.sym('U', nu, N)
        P = ca.SX.sym('P', nx + 3 + max_obs * 4)

        obj = 0
        g = [X[:, 0] - P[0:nx]]
        for k in range(N):
            x_next = X[:, k] + dt * f(X[:, k], U[:, k])
            g.append(X[:, k + 1] - x_next)
            pos_err = X[0:3, k] - P[nx:nx + 3]
            obj += self.Qp * ca.sumsqr(pos_err) + self.Qv * ca.sumsqr(X[3:6, k]) + self.R * ca.sumsqr(U[:, k])
        obj += self.Qp * ca.sumsqr(X[0:3, N] - P[nx:nx + 3])

        obs_start = nx + 3
        for k in range(N + 1):
            for j in range(max_obs):
                b = obs_start + 4 * j
                ox = P[b]
                oy = P[b + 1]
                oz = P[b + 2]
                orad = P[b + 3]
                dx = X[0, k] - ox
                dy = X[1, k] - oy
                dz = X[2, k] - oz
                g.append(dx * dx + dy * dy + dz * dz - (self.safe_radius + orad) ** 2)

        g_flat = ca.vertcat(*g)
        OPT = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        nlp = {'f': obj, 'x': OPT, 'g': g_flat, 'p': P}
        opts = {'ipopt.print_level': 0, 'print_time': 0,
                'ipopt.tol': 1e-3, 'ipopt.max_iter': 50,
                'ipopt.max_cpu_time': self.ipopt_time_limit,
                'ipopt.warm_start_init_point': "yes"}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        self.nx, self.nu, self.N = nx, nu, N
        self.n_opt = int(OPT.size1())
        self.g_dim = int(g_flat.size1())
        self.n_p = int(P.size1())
        self.obs_start = obs_start
        self.x_prev = np.zeros(self.n_opt, dtype=float)
        self.get_logger().info("[OK] MPC solver built successfully")

    # ------------------------------
    # Publisher loop
    # ------------------------------
    def pub_loop(self):
        self.latest_cmd.header.stamp = self.get_clock().now().to_msg()
        self.cmd_pub.publish(self.latest_cmd)

    # ------------------------------
    # Solver thread
    # ------------------------------
    def mpc_worker(self):
        nx, nu, N = self.nx, self.nu, self.N
        X_size = nx * (N + 1)
        U_size = nu * N

        while not self._stop_event.is_set() and rclpy.ok():
            if not self.odom_received:
                time.sleep(0.02)
                continue
            try:
                x0 = np.hstack([self.current_pos, self.current_vel])
                goal = self.goal
                obs_list = self.obstacles[:self.max_obs]
                obs_param = np.zeros(self.max_obs * 4, dtype=float)
                for i, o in enumerate(obs_list):
                    obs_param[i * 4:i * 4 + 4] = o
                for j in range(len(obs_list), self.max_obs):
                    obs_param[j * 4 + 3] = -self.safe_radius
                Pvec = np.concatenate([x0, goal, obs_param])

                lbx = -1e20 * np.ones(self.n_opt)
                ubx = 1e20 * np.ones(self.n_opt)
                for k in range(U_size):
                    lbx[X_size + k] = -self.max_acc
                    ubx[X_size + k] = self.max_acc

                g_l = np.zeros(self.g_dim)
                g_u = np.zeros(self.g_dim)
                g_l[:nx*(N+1)] = 0.0
                g_u[:nx*(N+1)] = 0.0
                g_l[nx*(N+1):] = 0.0
                g_u[nx*(N+1):] = 1e20

                x_init = self.x_prev if self.x_prev is not None else np.zeros(self.n_opt)
                try:
                    sol = self.solver(x0=x_init, p=Pvec, lbg=g_l, ubg=g_u, lbx=lbx, ubx=ubx)
                except Exception as e:
                    self.get_logger().warn(f"MPC solver exception: {e}")
                    sol = None

                if sol is None:
                    # fallback
                    brake_gain = 0.7
                    fallback_v = -brake_gain * self.current_vel
                    fallback_v = np.clip(fallback_v, -1.0, 1.0)
                    c, s = math.cos(self.current_yaw), math.sin(self.current_yaw)
                    v_body = np.array([c*fallback_v[0]+s*fallback_v[1],
                                       -s*fallback_v[0]+c*fallback_v[1],
                                       fallback_v[2]])
                    ts = TwistStamped()
                    ts.header.stamp = self.get_clock().now().to_msg()
                    ts.header.frame_id = 'base_link'
                    ts.twist.linear.x = float(v_body[0])
                    ts.twist.linear.y = float(v_body[1])
                    ts.twist.linear.z = float(v_body[2])
                    self.latest_cmd = ts
                    time.sleep(0.02)
                    continue

                sol_x = np.array(sol['x']).flatten()
                self.x_prev = sol_x.copy()
                # Extract first control
                U_opt = sol_x[X_size:X_size+U_size]
                a0 = U_opt[:3]
                v_world = self.current_vel + a0*self.dt
                v_world = np.clip(v_world, -2.0, 2.0)

                # Slowdown near goal
                err = np.linalg.norm(goal - self.current_pos)
                if err < 0.6:
                    v_world *= (err / 0.6)

                # ENU -> body
                c, s = math.cos(self.current_yaw), math.sin(self.current_yaw)
                v_body = np.array([c*v_world[0]+s*v_world[1], -s*v_world[0]+c*v_world[1], v_world[2]])

                ts = TwistStamped()
                ts.header.stamp = self.get_clock().now().to_msg()
                ts.header.frame_id = 'base_link'
                ts.twist.linear.x = float(v_body[0])
                ts.twist.linear.y = float(v_body[1])
                ts.twist.linear.z = float(v_body[2])
                self.latest_cmd = ts

                # --- Log min distance to obstacles ---
                if self.obstacles:
                    dists = [np.linalg.norm(self.current_pos - np.array(o[:3])) - o[3] for o in self.obstacles]
                    min_dist = min(dists)
                else:
                    min_dist = 999.0
                t_sec = self.get_clock().now().to_msg().sec
                self.get_logger().info(f"[INFO] pos={self.current_pos}, min_dist_obs={min_dist:.2f}")
                self.csv_writer.writerow([t_sec, *self.current_pos, *goal, min_dist])

                # Publish predicted trajectory (X positions)
                X_seq = []
                idx = 0
                sdim, udim = nx, nu
                X_seq.append(sol_x[idx:idx+sdim])
                idx += sdim
                for k in range(N):
                    idx += udim
                    X_seq.append(sol_x[idx:idx+sdim])
                    idx += sdim
                flat = []
                for xvec in X_seq:
                    flat.extend([float(xvec[0]), float(xvec[1]), float(xvec[2])])
                arr = Float32MultiArray()
                arr.data = flat
                self.traj_pub.publish(arr)

                time.sleep(0.02)

            except Exception as e:
                self.get_logger().warn(f"MPC outer exception: {e}")
                time.sleep(0.05)

    # ------------------------------
    # Utility
    # ------------------------------
    def quat_to_yaw(self, w, x, y, z):
        return math.atan2(2.0*(w*z+x*y), 1.0-2.0*(y*y+z*z))

    def destroy_node(self):
        self._stop_event.set()
        self.csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MPCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
