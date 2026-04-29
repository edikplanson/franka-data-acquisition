import time
from ros2_ws.src.tactile_pkg.tactile_pkg.tactile_sensor import TactileSensor

if __name__ == "__main__":
    sensor = TactileSensor("/dev/ttyUSB0", baud=2_000_000, rows=12, cols=32, thresh=20, noise_scale=60, init_frames=30, alpha=0.2)
    while True:
        sensor.update()
        sensor.visualizer() # visualize the current contact data
        data, timestamp = sensor.readSensor() # read the latest contact data and timestamp
        print(timestamp) 
        time.sleep(0.04) # 25 Hz update rate