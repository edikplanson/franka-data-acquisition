import cv2
import time

class Camera:
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        self.last_frame = None
        self.last_timestamp_ms = None
    def update(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        self.last_frame = frame
        self.last_timestamp_ms = int(time.time() * 1000)  # current time in milliseconds

    def readCam(self):
        return self.last_frame, self.last_timestamp_ms

    def release(self):
        self.cap.release()

    def visualizer(self):
        ret, frame = self.cap.read()
        if ret:
            cv2.imshow("Camera", frame)
            cv2.waitKey(1)
