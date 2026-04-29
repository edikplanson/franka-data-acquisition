import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class StateNode(Node):

    def __init__(self):
        super().__init__('state_node')

        self.pub = self.create_publisher(JointState, '/joint_states', 10)
        self.timer = self.create_timer(0.05, self.loop)

    def loop(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()

        # EXEMPLE FAKE (à remplacer par ton robot)
        msg.name = ["joint1", "joint2"]
        msg.position = [0.1, 0.2]

        self.pub.publish(msg)


def main():
    rclpy.init()
    node = StateNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()