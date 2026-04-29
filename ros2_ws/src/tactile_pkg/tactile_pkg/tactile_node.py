import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge

from tactile_sensor import TactileSensor


class TactileNode(Node):

    def __init__(self):
        super().__init__('tactile_node')

        self.pub = self.create_publisher(Image, '/tactile/image_raw', 10)
        self.bridge = CvBridge()

        self.sensor = TactileSensor(
            port="/dev/ttyUSB0",
            baud=2_000_000,
            rows=12,
            cols=32,
            thresh=20,
            noise_scale=60,
            init_frames=30,
            alpha=0.2
        )

        self.timer = self.create_timer(0.1, self.loop)

    def loop(self):
        self.sensor.update()
        frame, _ = self.sensor.readSensor()

        msg = self.bridge.cv2_to_imgmsg(frame, encoding="mono8")
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()

        self.pub.publish(msg)


def main():
    rclpy.init()
    node = TactileNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()