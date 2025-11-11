#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import PoseStamped, Twist
import numpy as np
import heapq
import math

class AStarPlanner(Node):
    def __init__(self):
        super().__init__('drone_astar')
        self.map = None
        self.map_received = False
        self.odom = None
        self.current_path = []
        self.waypoint_idx = 0
        self.reached_tolerance = 0.25  # mét

        # Publisher/Subscribers
        self.path_pub = self.create_publisher(Path, 'planned_path', 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(OccupancyGrid, '/map', self.map_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_cb, 10)

        # Timer loop
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info("🚀 Node A* đã khởi động thành công!")

    # ===== CALLBACKS =====
    def map_cb(self, msg):
        self.map = msg
        self.map_received = True
        self.get_logger().info(f"🗺️ Đã nhận map kích thước {msg.info.width}x{msg.info.height}")

    def odom_cb(self, msg):
        self.odom = msg

    def goal_cb(self, msg):
        if not self.map_received:
            self.get_logger().warn("⚠️ Chưa nhận map - không thể lập kế hoạch!")
            return
        start = self.get_current_xy()
        goal = (msg.pose.position.x, msg.pose.position.y)
        if start is None:
            self.get_logger().warn("⚠️ Chưa có dữ liệu odom hợp lệ!")
            return
        self.get_logger().info(f"🎯 Lập kế hoạch từ {start} đến {goal}")
        path = self.plan_astar(start, goal)
        if path:
            self.get_logger().info(f"✅ Đường đi tìm được có {len(path)} điểm")
            self.current_path = path
            self.waypoint_idx = 0
            self.publish_path(path)
        else:
            self.get_logger().error("❌ Không tìm được đường đi khả thi!")

    def get_current_xy(self):
        if self.odom is None:
            return None
        p = self.odom.pose.pose.position
        return (p.x, p.y)

    # ===== A* CORE =====
    def world_to_grid(self, x, y):
        origin = self.map.info.origin.position
        res = self.map.info.resolution
        gx = int((x - origin.x) / res)
        gy = int((y - origin.y) / res)
        return gx, gy

    def grid_to_world(self, gx, gy):
        origin = self.map.info.origin.position
        res = self.map.info.resolution
        x = gx * res + origin.x + res/2.0
        y = gy * res + origin.y + res/2.0
        return x, y

    def in_bounds(self, gx, gy):
        w = self.map.info.width
        h = self.map.info.height
        return 0 <= gx < w and 0 <= gy < h

    def is_free(self, gx, gy):
        idx = gy * self.map.info.width + gx
        if idx < 0 or idx >= len(self.map.data):
            return False
        val = self.map.data[idx]
        return val <= 50 and val >= -1  # -1 unknown, <=50 free

    def neighbors(self, gx, gy):
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = gx + dx, gy + dy
            if self.in_bounds(nx, ny) and self.is_free(nx, ny):
                yield nx, ny

    def heuristic(self, a, b):
        (ax, ay), (bx, by) = a, b
        return math.hypot(ax - bx, ay - by)

    def plan_astar(self, start_w, goal_w):
        start_g = self.world_to_grid(*start_w)
        goal_g = self.world_to_grid(*goal_w)

        if not self.in_bounds(*start_g) or not self.in_bounds(*goal_g):
            self.get_logger().error("⚠️ Start hoặc Goal nằm ngoài bản đồ!")
            return None

        open_set = []
        heapq.heappush(open_set, (0, start_g))
        came_from = {}
        gscore = {start_g: 0}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal_g:
                # reconstruct path
                path = []
                cur = current
                while cur in came_from:
                    path.append(cur)
                    cur = came_from[cur]
                path.append(start_g)
                path.reverse()
                world_path = [self.grid_to_world(gx, gy) for gx, gy in path]
                return world_path

            for nb in self.neighbors(*current):
                tentative_g = gscore[current] + self.heuristic(current, nb)
                if nb not in gscore or tentative_g < gscore[nb]:
                    came_from[nb] = current
                    gscore[nb] = tentative_g
                    f = tentative_g + self.heuristic(nb, goal_g)
                    heapq.heappush(open_set, (f, nb))
        return None

    # ===== PUBLISH PATH =====
    def publish_path(self, world_path):
        path_msg = Path()
        path_msg.header.frame_id = self.map.header.frame_id
        path_msg.header.stamp = self.get_clock().now().to_msg()
        for (x, y) in world_path:
            p = PoseStamped()
            p.header = path_msg.header
            p.pose.position.x = x
            p.pose.position.y = y
            path_msg.poses.append(p)
        self.path_pub.publish(path_msg)
        self.get_logger().info(f"📤 Đã gửi đường đi gồm {len(world_path)} điểm")

    # ===== SIMPLE FOLLOWER =====
    def control_loop(self):
        if not self.current_path or self.odom is None:
            return
        cur_pos = self.get_current_xy()
        if cur_pos is None:
            return
        target = self.current_path[self.waypoint_idx]
        dx = target[0] - cur_pos[0]
        dy = target[1] - cur_pos[1]
        dist = math.hypot(dx, dy)

        if dist < self.reached_tolerance:
            if self.waypoint_idx < len(self.current_path) - 1:
                self.waypoint_idx += 1
                self.get_logger().info(f"➡️ Chuyển tới waypoint {self.waypoint_idx}")
            else:
                self.get_logger().info("🏁 Đã đến đích!")
                self.current_path = []
                stop = Twist()
                self.cmd_pub.publish(stop)
                return

        vx = 0.5 * (dx / max(dist, 0.01))
        vy = 0.5 * (dy / max(dist, 0.01))

        cmd = Twist()
        cmd.linear.x = vx
        cmd.linear.y = vy
        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = AStarPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
