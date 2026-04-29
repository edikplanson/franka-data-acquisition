import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/eplanson/Documents/franka-data-acquisition/ros2_ws/install/my_robot'
