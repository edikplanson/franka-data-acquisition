import cv2
import time
import threading
import signal
import sys
import json
import numpy as np
import pandas as pd
import pyrealsense2 as rs
from pathlib import Path
from collections import deque

from tactile_sensor import TactileSensor

# =========================
# CONFIG
# =========================

FPS        = 30
DATA_ROOT  = Path("dataset")
TASK_NAME  = "teleoperation"
STATE_DIM  = 8    # 7 joints + gripper
ACTION_DIM = 7

RS_W,  RS_H  = 640,  480
AR_W,  AR_H  = 960,  600
TAC_W, TAC_H = 320,  120

# =========================
# TACTILE SENSOR
# =========================

sensor = TactileSensor(
    "/dev/ttyUSB0",
    baud=2_000_000,
    rows=12,
    cols=32,
    thresh=20,
    noise_scale=60,
    init_frames=30,
    alpha=0.8,
)

# =========================
# GLOBAL STATE
# =========================

lock           = threading.Lock()
tactile_buffer = deque(maxlen=3000)

latest: dict = {
    "realsense": None,
    "cam_ar1":   None,
    "cam_ar2":   None,
    "state":     np.zeros(STATE_DIM,  dtype=np.float32),
    "action":    np.zeros(ACTION_DIM, dtype=np.float32),
}

# Session-level accumulators (shared across episodes in one run)
_total_frames   = 0
_total_episodes = 0

# =========================
# DIRECTORY & PATH HELPERS
# =========================

def _file_id() -> int:
    """Auto-increment file ID to avoid overwriting previous sessions."""
    ref = DATA_ROOT / "videos" / "observation.images.realsense" / "chunk-000"
    if not ref.exists():
        return 0
    return len(list(ref.glob("file-*.mp4")))

FILE_ID  = None   # set once in main before recording starts
CHUNK_ID = 0


def video_path(cam: str) -> Path:
    return (DATA_ROOT / "videos" / f"observation.images.{cam}"
            / f"chunk-{CHUNK_ID:03d}" / f"file-{FILE_ID:03d}.mp4")


def parquet_data_path() -> Path:
    return DATA_ROOT / "data" / f"chunk-{CHUNK_ID:03d}" / f"file-{FILE_ID:03d}.parquet"


def episodes_meta_path() -> Path:
    return (DATA_ROOT / "meta" / "episodes"
            / f"chunk-{CHUNK_ID:03d}" / f"file-{FILE_ID:03d}.parquet")


def make_dirs():
    (DATA_ROOT / "data"   / f"chunk-{CHUNK_ID:03d}").mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / "meta"   / "episodes" / f"chunk-{CHUNK_ID:03d}").mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / "meta").mkdir(parents=True, exist_ok=True)
    for cam in ["realsense", "cam_ar1", "cam_ar2", "tactile"]:
        (DATA_ROOT / "videos" / f"observation.images.{cam}"
                   / f"chunk-{CHUNK_ID:03d}").mkdir(parents=True, exist_ok=True)

# =========================
# VIDEO WRITERS (session-level, stay open across episodes)
# =========================

def make_writer(path: Path, fps: int, w: int, h: int) -> cv2.VideoWriter:
    return cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

# Writers opened once per session in main(), closed on exit
writers: dict[str, cv2.VideoWriter] = {}

# =========================
# REALSENSE THREAD
# =========================

def realsense_thread(stop_evt: threading.Event):
    pipe = rs.pipeline()
    cfg  = rs.config()
    cfg.enable_stream(rs.stream.color, RS_W, RS_H, rs.format.bgr8, FPS)
    pipe.start(cfg)
    try:
        while not stop_evt.is_set():
            frames = pipe.wait_for_frames()
            frame  = frames.get_color_frame()
            if not frame:
                continue
            img = np.asanyarray(frame.get_data())
            writers["realsense"].write(img)
            with lock:
                latest["realsense"] = img
    finally:
        pipe.stop()

# =========================
# ARCAM THREAD
# =========================

def arcam_thread(dev: int, cam_key: str, stop_evt: threading.Event):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  AR_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, AR_H)
    cap.set(cv2.CAP_PROP_FPS,          FPS)
    try:
        while not stop_evt.is_set():
            ret, frame = cap.read()
            if not ret:
                continue
            writers[cam_key].write(frame)
            with lock:
                latest[cam_key] = frame
    finally:
        cap.release()

# =========================
# TACTILE THREAD
# =========================

def tactile_thread(stop_evt: threading.Event):
    while not stop_evt.is_set():
        t0 = time.time()
        sensor.update()
        img = sensor.get_colormap()
        img = cv2.resize(img, (TAC_W, TAC_H))
        ts  = time.time()
        writers["tactile"].write(img)
        with lock:
            tactile_buffer.append((ts, img))
        time.sleep(max(0.0, 1 / FPS - (time.time() - t0)))


def get_closest_tactile_ts(ts_target: float) -> float | None:
    if not tactile_buffer:
        return None
    return min(tactile_buffer, key=lambda x: abs(x[0] - ts_target))[0]

# =========================
# SYNCHRONIZER
# =========================

def synchronizer(stop_evt: threading.Event, frame_offset: int) -> list[dict]:
    """Records tabular data at FPS. Returns list of steps."""
    steps  = []
    t_next = time.time()
    fi     = frame_offset

    while not stop_evt.is_set():
        t_next += 1 / FPS

        with lock:
            state  = latest["state"].copy()
            action = latest["action"].copy()

        steps.append({
            "episode_index":      _total_episodes,
            "frame_index":        fi,
            "timestamp":          round(fi / FPS, 6),
            "index":              fi,
            "task_index":         0,
            "observation.state":  state.tolist(),
            "action":             action.tolist(),
        })
        fi += 1

        sleep = t_next - time.time()
        if sleep > 0:
            time.sleep(sleep)

    return steps

# =========================
# SAVE — PARQUET DATA
# =========================

def save_steps(steps: list[dict]):
    df_new = pd.DataFrame(steps)
    path   = parquet_data_path()
    if path.exists():
        df_old = pd.read_parquet(path)
        df_out = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_parquet(path, engine="pyarrow", index=False)

# =========================
# SAVE — EPISODE META
# =========================

def save_episode_meta(frame_offset: int, length: int):
    row  = {
        "episode_index":     _total_episodes,
        "task_index":        0,
        "length":            length,
        "data/chunk_index":  CHUNK_ID,
        "data/file_index":   FILE_ID,
        "frame_offset":      frame_offset,
    }
    path   = episodes_meta_path()
    df_new = pd.DataFrame([row])
    if path.exists():
        df_old = pd.read_parquet(path)
        df_out = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_parquet(path, engine="pyarrow", index=False)

# =========================
# SAVE — GLOBAL META
# =========================

def save_info():
    info = {
        "codebase_version": "v3.0",
        "robot_type":       "franka",
        "fps":              FPS,
        "total_episodes":   _total_episodes,
        "total_frames":     _total_frames,
        "total_tasks":      1,
        "splits":           {"train": f"0:{_total_episodes}"},
        "data_path":        f"data/chunk-{{chunk_index:03d}}/file-{{file_index:03d}}.parquet",
        "video_path":       "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.images.realsense": {
                "dtype": "video", "shape": [RS_H, RS_W, 3],
                "video_info": {"video.fps": FPS, "video.codec": "mp4v"},
            },
            "observation.images.cam_ar1": {
                "dtype": "video", "shape": [AR_H, AR_W, 3],
                "video_info": {"video.fps": FPS, "video.codec": "mp4v"},
            },
            "observation.images.cam_ar2": {
                "dtype": "video", "shape": [AR_H, AR_W, 3],
                "video_info": {"video.fps": FPS, "video.codec": "mp4v"},
            },
            "observation.images.tactile": {
                "dtype": "video", "shape": [TAC_H, TAC_W, 3],
                "video_info": {"video.fps": FPS, "video.codec": "mp4v"},
            },
            "observation.state": {
                "dtype": "float32", "shape": [STATE_DIM],
                "names": [f"joint_{i}" for i in range(STATE_DIM)],
            },
            "action": {
                "dtype": "float32", "shape": [ACTION_DIM],
                "names": [f"joint_{i}" for i in range(ACTION_DIM)],
            },
        },
    }
    with open(DATA_ROOT / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)


def save_tasks():
    path = DATA_ROOT / "meta" / "tasks.jsonl"
    with open(path, "w") as f:
        f.write(json.dumps({"task_index": 0, "task": TASK_NAME}) + "\n")


def compute_stats():
    path = parquet_data_path()
    if not path.exists():
        return
    df = pd.read_parquet(path)

    def stats(col):
        mat = np.stack(df[col].values)
        return {"mean": mat.mean(0).tolist(), "std": mat.std(0).tolist(),
                "min":  mat.min(0).tolist(),  "max": mat.max(0).tolist()}

    out = {"observation.state": stats("observation.state"),
           "action":            stats("action")}
    with open(DATA_ROOT / "meta" / "stats.json", "w") as f:
        json.dump(out, f, indent=2)

# =========================
# EPISODE
# =========================

def run_episode():
    global _total_frames, _total_episodes

    frame_offset  = _total_frames
    episode_index = _total_episodes

    stop_evt = threading.Event()

    threads = [
        threading.Thread(target=realsense_thread, args=(stop_evt,),        daemon=True),
        threading.Thread(target=arcam_thread,     args=(0, "cam_ar1", stop_evt), daemon=True),
        threading.Thread(target=arcam_thread,     args=(2, "cam_ar2", stop_evt), daemon=True),
        threading.Thread(target=tactile_thread,   args=(stop_evt,),        daemon=True),
    ]

    steps_container = [None]

    def sync_worker():
        steps_container[0] = synchronizer(stop_evt, frame_offset)

    sync_thread = threading.Thread(target=sync_worker, daemon=True)
    threads.append(sync_thread)

    for t in threads:
        t.start()

    print(f"\n● Recording episode {episode_index:06d}  (offset={frame_offset})")
    input("  Press ENTER to stop...\n")

    stop_evt.set()
    for t in threads:
        t.join(timeout=3)

    # =========================
    # SAVE (near-instant: videos already on disk)
    # =========================

    steps  = steps_container[0] or []
    length = len(steps)

    if length > 0:
        t0 = time.time()
        save_steps(steps)
        save_episode_meta(frame_offset, length)
        _total_frames   += length
        _total_episodes += 1
        save_info()
        save_tasks()
        compute_stats()
        print(f"✔ Episode {episode_index:06d} saved — {length} frames in {time.time()-t0:.2f}s")
    else:
        print("⚠ Empty episode skipped")

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    FILE_ID = _file_id()
    make_dirs()

    writers["realsense"] = make_writer(video_path("realsense"), FPS, RS_W,  RS_H)
    writers["cam_ar1"]   = make_writer(video_path("cam_ar1"),   FPS, AR_W,  AR_H)
    writers["cam_ar2"]   = make_writer(video_path("cam_ar2"),   FPS, AR_W,  AR_H)
    writers["tactile"]   = make_writer(video_path("tactile"),   FPS, TAC_W, TAC_H)

    print(f"=== Franka recorder  (file-{FILE_ID:03d}) ===")
    print(f"Dataset: {DATA_ROOT.resolve()}")

    def _shutdown(*_):
        print("\nClosing writers...")
        for w in writers.values():
            w.release()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)

    try:
        while True:
            input(f"\nPress ENTER to start episode {_total_episodes:06d}...")
            run_episode()
    finally:
        for w in writers.values():
            w.release()
