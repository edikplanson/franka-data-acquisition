import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, JointState
from message_filters import Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge

import os
import cv2
import json


class DatasetLogger(Node):

    def __init__(self):
        super().__init__('dataset_logger')

        self.bridge = CvBridge()

        self.tactile = Subscriber(self, Image, '/tactile/image_raw')
        self.camera = Subscriber(self, Image, '/camera/image_raw')
        self.state = Subscriber(self, JointState, '/joint_states')

        self.sync = ApproximateTimeSynchronizer(
            [self.tactile, self.camera, self.state],
            queue_size=20,
            slop=0.05
        )

        self.sync.registerCallback(self.callback)

        self.base_dir = "data/raw"
        os.makedirs(self.base_dir, exist_ok=True)

        self.index = 0

        self.get_logger().info("Dataset logger started")

    def callback(self, tactile_msg, cam_msg, state_msg):

        self.index += 1
        folder = os.path.join(self.base_dir, f"{self.index:06d}")
        os.makedirs(folder, exist_ok=True)

        # tactile
        t = self.bridge.imgmsg_to_cv2(tactile_msg, "mono8")
        cv2.imwrite(f"{folder}/tactile.png", t)

        # camera
        c = self.bridge.imgmsg_to_cv2(cam_msg, "bgr8")
        cv2.imwrite(f"{folder}/camera.png", c)

        # state
        state_dict = {
            "names": list(state_msg.name),
            "positions": list(state_msg.position)
        }

        with open(f"{folder}/state.json", "w") as f:
            json.dump(state_dict, f)

        # timestamp
        with open(f"{folder}/timestamp.txt", "w") as f:
            f.write(str(tactile_msg.header.stamp))


def main():
    rclpy.init()
    node = DatasetLogger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()