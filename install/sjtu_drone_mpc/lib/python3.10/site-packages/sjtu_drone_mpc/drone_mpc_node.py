
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import jax.numpy as jnp
from pyNMPC.nmpc import NMPC, MPCParams

class DroneMPC(Node):
    def __init__(self):
        super().__init__('drone_mpc')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.timer = self.create_timer(0.1, self.control_loop)
        self.state = jnp.array([0.0, 0.0, 0.0])
        self.ref = jnp.array([2.0, 2.0, 0.0])

        def dynamics(x, u, dt):
            theta = x[2]
            return x + jnp.array([
                u[0]*jnp.cos(theta),
                u[0]*jnp.sin(theta),
                u[1]
            ]) * dt

        params = MPCParams(
            dt=0.1,
            N=15,
            n_states=3,
            n_controls=2,
            Q=jnp.diag(jnp.array([10., 10., 1.])),
            QN=jnp.diag(jnp.array([10., 10., 1.])),
            R=jnp.diag(jnp.array([1., 0.1])),
            x_ref=self.ref,
            x_min=jnp.array([-10, -10, -jnp.inf]),
            x_max=jnp.array([10, 10, jnp.inf]),
            u_min=jnp.array([0., -1.0]),
            u_max=jnp.array([1.0, 1.0]),
            slack_weight=1e4,
        )

        self.mpc = NMPC(dynamics_fn=dynamics, params=params)
        self.get_logger().info("MPC node ready")

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        # nếu odom chứa yaw, bạn có thể lấy orientation -> yaw
        self.state = jnp.array([pos.x, pos.y, 0.0])

    def control_loop(self):
        try:
            result = self.mpc.solve(x0=self.state, x_ref=self.ref)
            u = result.u_traj[0]
            msg = Twist()
            msg.linear.x = float(u[0])
            msg.angular.z = float(u[1])
            self.pub.publish(msg)
        except Exception as e:
            self.get_logger().warning(f"MPC error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = DroneMPC()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
