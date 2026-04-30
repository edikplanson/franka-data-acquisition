import cv2
import serial
import threading
import numpy as np
import time

# -------------------------------------------------------------------------
# Inspired from FlexiTac Hardware Repository
# Author: Binghao Huang (binghao-huang / Binghao_Huang)
# https://github.com/FlexiTac/FlexiTac_Hardware_Repo
#
# Used as reference for tactile sensor acquisition and processing pipeline.
# -------------------------------------------------------------------------

class TactileSensor:
    def __init__(self,port,baud,rows,cols,thresh,noise_scale,init_frames,alpha): 
        # port, baudrate, rows, cols, threshold for contact, noise scale for normalization, number of frames to initialize baseline, alpha for temporal filtering
        self.PORT = port
        self.BAUD = baud
        self.ROWS = rows
        self.COLS = cols
        self.FRAME_BYTES = self.ROWS * self.COLS 
        self.THRESHOLD = thresh
        self.NOISE_SCALE = noise_scale
        self.INIT_FRAMES = init_frames
        self.ALPHA = alpha
        self.MAGIC = b"\xAA\x55"
    
        self.serDev = serial.Serial(self.PORT, self.BAUD)
        self.serialThread = None
        self.contact_data_norm = np.zeros((self.ROWS, self.COLS), dtype=np.float32)
        self.prev_frame = np.zeros_like(self.contact_data_norm, dtype=np.float32)
        self.flag=False
        self.temp_filtered_data = np.zeros_like(self.contact_data_norm, dtype=np.float32)
        self.temp_filtered_data_scaled = np.zeros_like(self.contact_data_norm, dtype=np.uint8)
        self.timestamp_ms = 0
        self.setup()

    def setup(self): # flush serial buffer and start thread
        self.serDev.flush()
        self.serDev.reset_input_buffer()

        self.serialThread = threading.Thread(target=self.readThread, args=(self.serDev,))
        self.serialThread.daemon = True
        self.serialThread.start()   

    def readThread(self,serDev): # read serial data, initialize baseline, and update contact_data_norm in a loop
        ring = bytearray()
        frame_buf = bytearray(self.FRAME_BYTES)
        data_tac = []
        t1 = time.time()
        self.flag = False
        serDev.timeout = 0.01 # helps responsiveness
        def read_exact(n): 
            """Read exactly n bytes from serial (blocking up to timeout loops)."""
            buf = bytearray(n)
            mv = memoryview(buf)
            got = 0
            while got < n:
                r = serDev.readinto(mv[got:])
                if r is None:
                    r = 0
                if r == 0:
                    return None
                got += r
            return buf
        # Init loop: collect INIT_FRAMES frames to compute median baseline
        while True:
            chunk = serDev.read(4096)
            if not chunk:
                continue
            ring.extend(chunk)

            # keep buffer bounded
            if len(ring) > 50000:
                ring = ring[-50000:]

            idx = ring.find(self.MAGIC)
            if idx < 0:
                # keep last byte in case marker splits
                if len(ring) > 1:
                    ring = ring[-1:]
                continue

            # drop before marker, consume marker
            if idx > 0:
                del ring[:idx]
            if len(ring) < 2:
                continue
            del ring[:2]

            # read one frame (512)
            if len(ring) >= self.FRAME_BYTES:
                frame_buf[:] = ring[:self.FRAME_BYTES]
                del ring[:self.FRAME_BYTES]
            else:
                have = len(ring)
                frame_buf[:have] = ring[:have]
                del ring[:have]
                rem = self.FRAME_BYTES - have
                rest = read_exact(rem)
                if rest is None:
                    ring.clear()
                    continue
                frame_buf[have:] = rest

            frame = np.frombuffer(frame_buf, dtype=np.uint8).reshape((self.ROWS, self.COLS)).astype(np.float32)
            data_tac.append(frame)

            now = time.time()
            print("init fps", 1 / (now - t1 + 1e-9))
            t1 = time.time()

            if len(data_tac) >= self.INIT_FRAMES:
                break

        data_tac = np.stack(data_tac, axis=0)  # (N,16,32)
        median = np.median(data_tac, axis=0)
        self.flag = True
        print("Finish Initialization!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        while True: # main loop to read frames and update contact_data_norm
            chunk = serDev.read(8192)
            if not chunk:
                continue
            ring.extend(chunk)

            if len(ring) > 50000:
                ring = ring[-50000:]

            # parse as many complete frames as available
            while True:
                idx = ring.find(self.MAGIC)
                if idx < 0:
                    # keep last byte for split marker
                    if len(ring) > 1:
                        ring = ring[-1:]
                    break

                # drop before marker
                if idx > 0:
                    del ring[:idx]

                # need marker + frame
                if len(ring) < 2 + self.FRAME_BYTES:
                    # not enough yet
                    break

                # consume marker
                del ring[:2]

                # take frame bytes
                frame_bytes = ring[:self.FRAME_BYTES]
                del ring[:self.FRAME_BYTES]

                backup = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((self.ROWS, self.COLS)).astype(np.float32)

                # old-style processing
                contact_data = backup - median - self.THRESHOLD
                contact_data = np.clip(contact_data, 0, 100)

                if np.max(contact_data) < self.THRESHOLD:
                    self.contact_data_norm = contact_data / self.NOISE_SCALE
                else:
                    self.contact_data_norm = contact_data / (np.max(contact_data) + 1e-6)

    def temporal_filter(self, new_frame, prev_frame): # simple exponential moving average filter
        return self.ALPHA * new_frame + (1 - self.ALPHA) * prev_frame
    def get_colormap(self):
        return cv2.applyColorMap(self.temp_filtered_data_scaled, cv2.COLORMAP_VIRIDIS)
    def visualizer(self):
        colormap = self.get_colormap()
        cv2.imshow("Contact Data_left", colormap)
        cv2.waitKey(1)

    def update(self): # apply temporal filter to contact_data_norm and update temp_filtered_data and temp_filtered_data_scaled
        if self.flag:
            self.temp_filtered_data = self.temporal_filter(self.contact_data_norm, self.prev_frame)
            self.prev_frame = self.temp_filtered_data
            self.temp_filtered_data_scaled = np.clip(self.temp_filtered_data * 255.0, 0, 255).astype(np.uint8)
            self.timestamp_ms = int(time.time() * 1000) # update timestamp in milliseconds

    def readSensor(self): # return the current filtered and scaled contact data
        return self.temp_filtered_data_scaled, self.timestamp_ms
    
    def readTaxel(self,row,col): # return the value of a specific taxel from the filtered and scaled contact data
        return self.temp_filtered_data_scaled[row,col]