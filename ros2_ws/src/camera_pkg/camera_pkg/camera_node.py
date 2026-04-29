import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class CameraNode(Node):

    def __init__(self):
        super().__init__('camera_node')

        self.pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.bridge = CvBridge()

        self.cap = cv2.VideoCapture(0)

        self.timer = self.create_timer(0.033, self.loop)  # ~30 FPS

    def loop(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()

        self.pub.publish(msg)


def main():
    rclpy.init()
    node = CameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()