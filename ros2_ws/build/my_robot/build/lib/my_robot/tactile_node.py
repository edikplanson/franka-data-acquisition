import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
import cv2
import numpy as np
from cv_bridge import CvBridge

from my_robot.tactile_reader import TactileSensor

class TactileNode(Node):
    def __init__(self):
        super().__init__('tactile_node')

        # Paramètres ROS 2 (modifiables sans recompiler)
        self.declare_parameter('port', '/dev/ttyUSB0')
        self.declare_parameter('baud', 2_000_000)
        self.declare_parameter('rows', 12)
        self.declare_parameter('cols', 32)
        self.declare_parameter('thresh', 20)
        self.declare_parameter('noise_scale', 60)
        self.declare_parameter('init_frames', 30)
        self.declare_parameter('alpha', 0.2)

        port         = self.get_parameter('port').value
        baud         = self.get_parameter('baud').value
        rows         = self.get_parameter('rows').value
        cols         = self.get_parameter('cols').value
        thresh       = self.get_parameter('thresh').value
        noise_scale  = self.get_parameter('noise_scale').value
        init_frames  = self.get_parameter('init_frames').value
        alpha        = self.get_parameter('alpha').value

        self.sensor = TactileSensor(port, baud, rows, cols, thresh, noise_scale, init_frames, alpha)
        self.bridge = CvBridge()

        # Publisher image brute (uint8, visualisable avec rqt_image_view)
        self.pub_image = self.create_publisher(Image, 'tactile/image', 10)

        # Timer à 25 Hz
        self.create_timer(0.04, self.timer_callback)
        self.get_logger().info('Tactile node démarré, en attente d\'initialisation...')

    def timer_callback(self):
        self.sensor.update()

        data, timestamp_ms = self.sensor.readSensor()  # uint8 (rows, cols)

        if not self.sensor.flag:
            return  # pas encore initialisé

        # Colormap pour la visualisation
        colormap = cv2.applyColorMap(data, cv2.COLORMAP_VIRIDIS)  # BGR uint8

        # Construire le message Image
        msg = self.bridge.cv2_to_imgmsg(colormap, encoding='bgr8')
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'tactile_sensor'

        self.pub_image.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = TactileNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()