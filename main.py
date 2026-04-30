import pyrealsense2 as rs
import cv2
import threading
import time
import signal
import sys
from pathlib import Path
import numpy as np

from tactile_sensor import TactileSensor

# =========================
# INIT TACTILE
# =========================
sensor = TactileSensor(
    "/dev/ttyUSB0",
    baud=2_000_000,
    rows=12,
    cols=32,
    thresh=20,
    noise_scale=60,
    init_frames=30,
    alpha=0.2
)

FPS_AR = 30
DATA_ROOT = Path("data")
DATA_ROOT.mkdir(exist_ok=True)


def get_next_session_id():
    existing = sorted([
        d for d in DATA_ROOT.iterdir()
        if d.is_dir() and d.name.isdigit()
    ])
    return int(existing[-1].name) + 1 if existing else 1


def make_session_dirs(session_id: int):
    base = DATA_ROOT / f"{session_id:05d}"
    dirs = {
        "realsense": base / "cam_realsense",
        "ar0234_1":  base / "cam_ar_1",
        "ar0234_2":  base / "cam_ar_2",
        "tactile":   base / "cap_tactile",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# =========================
# CAPTURE THREADS
# =========================

def capture_realsense(output_dirs, start_evt, stop_evt, barrier):
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipe.start(cfg)

    idx = 0
    try:
        start_evt.wait()
        while not stop_evt.is_set():
            barrier.wait()
            frames = pipe.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue
            ts = time.time()
            img = np.asanyarray(color.get_data())
            fname = output_dirs["realsense"] / f"{idx:06d}_{ts:.6f}.png"
            cv2.imwrite(str(fname), img)
            idx += 1
    finally:
        pipe.stop()


def capture_arducam(dev_id: int, key: str, fps: int, output_dirs, start_evt, stop_evt, barrier):
    cap = cv2.VideoCapture(dev_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)

    idx = 0
    try:
        start_evt.wait()
        while not stop_evt.is_set():
            barrier.wait()
            ret, frame = cap.read()
            if not ret:
                continue
            ts = time.time()
            fname = output_dirs[key] / f"{idx:06d}_{ts:.6f}.png"
            cv2.imwrite(str(fname), frame)
            idx += 1
    finally:
        cap.release()


def capture_tactile(sensor, output_dirs, start_evt, stop_evt, barrier):
    idx = 0
    try:
        start_evt.wait()
        while not stop_evt.is_set():
            barrier.wait()
            sensor.update()
            img = sensor.get_colormap()
            ts = time.time()
            fname = output_dirs["tactile"] / f"{idx:06d}_{ts:.6f}.png"
            cv2.imwrite(str(fname), img)
            idx += 1
    except Exception as e:
        print("Tactile thread error:", e)


# =========================
# WARM-UP
# =========================

def warmup_cameras():
    print("Warming up cameras...")

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipe.start(cfg)
    for _ in range(30):
        pipe.wait_for_frames()
    pipe.stop()

    cap0 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(2)
    for _ in range(30):
        cap0.read()
        cap2.read()
    cap0.release()
    cap2.release()

    print("Cameras ready.")


# =========================
# SESSION
# =========================

def run_session(session_id: int):
    output_dirs = make_session_dirs(session_id)
    session_path = DATA_ROOT / f"{session_id:05d}"

    start_evt = threading.Event()
    stop_evt  = threading.Event()
    barrier   = threading.Barrier(4)

    threads = [
        threading.Thread(
            target=capture_realsense,
            args=(output_dirs, start_evt, stop_evt, barrier),
            daemon=True
        ),
        threading.Thread(
            target=capture_arducam,
            args=(0, "ar0234_1", FPS_AR, output_dirs, start_evt, stop_evt, barrier),
            daemon=True
        ),
        threading.Thread(
            target=capture_arducam,
            args=(2, "ar0234_2", FPS_AR, output_dirs, start_evt, stop_evt, barrier),
            daemon=True
        ),
        threading.Thread(
            target=capture_tactile,
            args=(sensor, output_dirs, start_evt, stop_evt, barrier),
            daemon=True
        ),
    ]

    for t in threads:
        t.start()

    t_start = time.time()
    start_evt.set()

    print(f"\n  Recording → {session_path}")
    print("  Press ENTER to stop...\n")
    input()

    stop_evt.set()
    t_end = time.time()

    for t in threads:
        t.join(timeout=5)

    duration = t_end - t_start

    # --- bilan ---
    total_frames = sum(
        len(list(d.glob("*.png"))) for d in output_dirs.values()
    )

    print(f"\n{'─'*40}")
    print(f"  Session #{session_id:05d} — bilan")
    print(f"{'─'*40}")
    print(f"  Durée        : {duration:.2f} s  ({duration/60:.1f} min)")
    print(f"  Dossier      : {session_path}")
    for key, d in output_dirs.items():
        n = len(list(d.glob("*.png")))
        fps_real = n / duration if duration > 0 else 0
        print(f"  {key:<12}: {n} frames  ({fps_real:.1f} fps)")
    print(f"  Total frames : {total_frames}")
    print(f"{'─'*40}\n")


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    # Ctrl+C propre
    signal.signal(signal.SIGINT, lambda *_: (print("\n\nArrêt demandé. Bye!"), sys.exit(0)))

    warmup_cameras()
    print("\nAll systems ready.")

    while True:
        session_id = get_next_session_id()
        print(f"\nProchaine session : #{session_id:05d}")
        print("Press ENTER to START recording...  (Ctrl+C to quit)")
        try:
            input()
        except KeyboardInterrupt:
            print("\nBye!")
            break

        try:
            run_session(session_id)
        except KeyboardInterrupt:
            print("\nSession interrompue.")
            break