import time
from tactile_reader import TactileSensor

if __name__ == "__main__":
    sensor = TactileSensor("/dev/ttyUSB0", 2_000_000, 12, 32, 20, 60, 30, 0.2)
    while True:
        sensor.update()
        sensor.visualizer() # visualize the current contact data
        print(sensor.readSensor()[1]) # print timestamp of the latest frame
        time.sleep(0.01)