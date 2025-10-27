#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Twist
import numpy as np
import heapq
import math
import time

class AStarPlanner(Node):
    def __init__(self):
        super().__init__('drone_astar')
        self.map = None
        self.map_received = False
        self.odom = None
        self.path_pub = self.create_publisher(Path, 'planned_path', 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)  # đổi nếu cần
        self.create_subscription(OccupancyGrid, '/map', self.map_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_cb, 10)
        self.timer = self.create_timer(0.1, self.control_loop)
        self.current_path = []
        self.waypoint_idx = 0
        self.reached_tolerance = 0.25  # m

    def map_cb(self, msg: OccupancyGrid):
        self.map = msg
        self.map_received = True

    def odom_cb(self, msg: Odometry):
        self.odom = msg

    def goal_cb(self, msg: PoseStamped):
        if not self.map_received:
            self.get_logger().warn("Chưa nhận map - không thể lập kế hoạch")
            return
        start = self.get_current_xy()
        goal = (msg.pose.position.x, msg.pose.position.y)
        if start is None:
            self.get_logger().warn("Chưa có odom hợp lệ")
            return
        self.get_logger().info(f"Planning from {start} to {goal}")
        path = self.plan_astar(start, goal)
        if path:
            self.current_path = path
            self.waypoint_idx = 0
            self.publish_path(path)
        else:
            self.get_logger().error("Không tìm được đường đi")

    def get_current_xy(self):
        if self.odom is None:
            return None
        p = self.odom.pose.pose.position
        return (p.x, p.y)

    # ---------- A* on occupancy grid ----------
    def world_to_grid(self, x, y):
        # occupancy grid: origin at map.info.origin (Pose), resolution
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
        val = self.map.data[idx]
        return val == -1 or val == 0  # -1 unknown; 0 free; >50 occupied

    def neighbors(self, gx, gy):
        # 4-connected
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = gx + dx, gy + dy
            if self.in_bounds(nx, ny) and self.is_free(nx, ny):
                yield nx, ny

    def heuristic(self, a, b):
        (ax, ay), (bx, by) = a, b
        return math.hypot(ax-bx, ay-by)

    def plan_astar(self, start_w, goal_w):
        # convert to grid
        start_g = self.world_to_grid(*start_w)
        goal_g = self.world_to_grid(*goal_w)
        w = self.map.info.width
        h = self.map.info.height

        open_set = []
        heapq.heappush(open_set, (0, start_g))
        came_from = {}
        gscore = {start_g: 0}
        fscore = {start_g: self.heuristic(start_w, goal_w)}

        max_iters = w*h
        iters = 0
        while open_set and iters < max_iters:
            iters += 1
            _, current = heapq.heappop(open_set)
            if current == goal_g:
                # reconstruct path
                path = []
                cur = current
                while cur != start_g:
                    path.append(cur)
                    cur = came_from[cur]
                path.append(start_g)
                path.reverse()
                # convert to world coordinates
                world_path = [self.grid_to_world(gx, gy) for (gx, gy) in path]
                return world_path
            for nb in self.neighbors(*current):
                tentative_g = gscore[current] + self.heuristic(self.grid_to_world(*current), self.grid_to_world(*nb))
                if nb not in gscore or tentative_g < gscore[nb]:
                    came_from[nb] = current
                    gscore[nb] = tentative_g
                    f = tentative_g + self.heuristic(self.grid_to_world(*nb), self.grid_to_world(*goal_g))
                    if nb not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f, nb))
        return None

    def publish_path(self, world_path):
        path_msg = Path()
        path_msg.header.frame_id = self.map.header.frame_id
        path_msg.header.stamp = self.get_clock().now().to_msg()
        for (x, y) in world_path:
            p = PoseStamped()
            p.header = path_msg.header
            p.pose.position.x = x
            p.pose.position.y = y
            p.pose.position.z = 0.0
            path_msg.poses.append(p)
        self.path_pub.publish(path_msg)
        self.get_logger().info(f"Published path with {len(world_path)} points")

    # ---------- simple waypoint follower ----------
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

        # if close -> next waypoint
        if dist < self.reached_tolerance:
            if self.waypoint_idx < len(self.current_path)-1:
                self.waypoint_idx += 1
                self.get_logger().info(f"Next waypoint {self.waypoint_idx}")
            else:
                self.get_logger().info("Đã đến đích")
                self.current_path = []
                self.waypoint_idx = 0
                # stop
                stop = Twist()
                self.cmd_pub.publish(stop)
                return

        # publish velocity towards waypoint (very simple P controller)
        vx = 0.6 * (dx/dist) if dist > 0.01 else 0.0
        vy = 0.6 * (dy/dist) if dist > 0.01 else 0.0

        cmd = Twist()
        cmd.linear.x = vx
        cmd.linear.y = vy
        cmd.linear.z = 0.0
        # optionally set yaw rate
        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = AStarPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
