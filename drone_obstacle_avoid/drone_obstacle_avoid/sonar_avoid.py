import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Range
from geometry_msgs.msg import Twist

class SonarAvoider(Node):
    def __init__(self):
        super().__init__('sonar_avoider')
        self.get_logger().info("🚁 Node tránh vật cản bằng sonar (Range) đã khởi động!")

        # Subscriber - nhận dữ liệu sonar
        self.sonar_sub = self.create_subscription(
            Range,
            '/simple_drone/sonar/out',
            self.sonar_callback,
            10)

        # Publisher - gửi lệnh điều khiển vận tốc
        self.pub = self.create_publisher(Twist, '/simple_drone/cmd_vel', 10)

    def sonar_callback(self, msg: Range):
        try:
            distance = msg.range
            cmd = Twist()

            # In khoảng cách để debug
            self.get_logger().info(f"📏 Khoảng cách vật cản: {distance:.2f} m")

            # Nếu vật cản gần hơn 1m → lùi lại
            if distance < 1.0:
                cmd.linear.x = -0.3
                self.get_logger().warn(f"⚠️ Vật cản gần {distance:.2f} m, lùi lại!")
            else:
                cmd.linear.x = 0.5
                self.get_logger().info(f"✅ Khoảng cách an toàn {distance:.2f} m, tiếp tục bay.")

            # Gửi lệnh ra topic điều khiển
            self.pub.publish(cmd)

        except Exception as e:
            self.get_logger().error(f"Lỗi trong sonar_callback: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = SonarAvoider()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
