"""
record.py — Franka VLA dataset recorder (format openpi / pi0.5-droid)

Cameras : RealSense 435i  → wrist_image_left       (poignet)
          ArduCam AR0234  → exterior_image_1_left   (vue extérieure gauche)
          ArduCam AR0234  → exterior_image_2_left   (vue extérieure droite)
Robot   : Franka Panda via panda-py
           joint_position  (7,)  — positions q
           gripper_position (1,) — largeur pince
           actions          (8,) — vitesses dq (7) + position pince (1)
Format  : LeRobot v3.0 — data/ meta/ videos/
"""

import cv2, time, threading, signal, sys, json
import numpy as np
import pandas as pd
import pyrealsense2 as rs
from pathlib import Path

# ── franka ────────────────────────────────────────────────────────────────────
FRANKA_IP = "172.17.1.2"
try:
    import panda_py
    _panda   = panda_py.Panda(FRANKA_IP)
    _gripper = panda_py.libfranka.Gripper(FRANKA_IP)
    print("Franka connected.")
except Exception as _e:
    _panda = _gripper = None
    print(f"[WARN] Franka not connected ({_e}) — state will be zeros.")

# ── config ────────────────────────────────────────────────────────────────────
FPS          = 15
DATA_ROOT    = Path("dataset")
AR_DEVS      = [0, 2]         # /dev/videoX pour les deux ArduCams
CHUNK_ID     = 0

# résolutions de capture natives (les frames sont ensuite redimensionnées)
RS_W,  RS_H  = 640, 480
AR_W,  AR_H  = 960, 600
# résolution de stockage imposée par openpi
IMG_W, IMG_H = 320, 180
# frames ignorées au début de chaque épisode (caméras vertes / instables)
WARMUP_FRAMES = 14

# noms de caméras attendus par openpi/pi0.5-droid
CAMERAS = ["wrist_image_left", "exterior_image_1_left", "exterior_image_2_left"]

# ── shared state ──────────────────────────────────────────────────────────────
_lock   = threading.Lock()
_latest: dict = {
    "wrist_image_left":      None,
    "exterior_image_1_left": None,
    "exterior_image_2_left": None,
    "joint_position":        np.zeros(7, np.float32),
    "joint_velocity":        np.zeros(7, np.float32),
    "gripper_position":      np.float32(0.0),
}
_total_frames   = 0
_total_episodes = 0
_tasks: dict[str, int] = {}
FILE_ID: int = 0

# ── path helpers ──────────────────────────────────────────────────────────────
def _vid_path(cam: str) -> Path:
    return DATA_ROOT / "videos" / cam / f"chunk-{CHUNK_ID:03d}" / f"file-{FILE_ID:03d}.mp4"

def _parquet_path() -> Path:
    return DATA_ROOT / "data" / f"chunk-{CHUNK_ID:03d}" / f"file-{FILE_ID:03d}.parquet"

def _episodes_path() -> Path:
    return DATA_ROOT / "meta" / "episodes" / f"chunk-{CHUNK_ID:03d}" / f"file-{FILE_ID:03d}.parquet"

def _init_file_id() -> int:
    ref = DATA_ROOT / "videos" / "wrist_image_left" / f"chunk-{CHUNK_ID:03d}"
    return len(list(ref.glob("file-*.mp4"))) if ref.exists() else 0

def _task_idx(instruction: str) -> int:
    if instruction not in _tasks:
        _tasks[instruction] = len(_tasks)
    return _tasks[instruction]

def _make_dirs():
    for cam in CAMERAS:
        _vid_path(cam).parent.mkdir(parents=True, exist_ok=True)
    _parquet_path().parent.mkdir(parents=True, exist_ok=True)
    _episodes_path().parent.mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / "meta").mkdir(parents=True, exist_ok=True)

# ── capture threads ───────────────────────────────────────────────────────────
def _realsense_thread(stop: threading.Event):
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.color, RS_W, RS_H, rs.format.bgr8, 30)
    pipe.start(cfg)
    try:
        while not stop.is_set():
            try:
                f = pipe.wait_for_frames(timeout_ms=5000).get_color_frame()
            except RuntimeError:
                continue
            if f:
                with _lock:
                    _latest["wrist_image_left"] = np.asanyarray(f.get_data())
    finally:
        pipe.stop()

def _arcam_thread(dev: int, key: str, stop: threading.Event):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  AR_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, AR_H)
    cap.set(cv2.CAP_PROP_FPS,          30)
    try:
        while not stop.is_set():
            ok, frame = cap.read()
            if ok:
                with _lock:
                    _latest[key] = frame
    finally:
        cap.release()

def _franka_thread(stop: threading.Event):
    """Lit q et dq directement depuis libfranka via panda-py (~100 Hz)."""
    if _panda is None:
        return
    while not stop.is_set():
        try:
            s  = _panda.get_state()
            gw = float(_gripper.read_once().width) if _gripper else 0.0
            with _lock:
                _latest["joint_position"]  = np.array(s.q,  np.float32)
                _latest["joint_velocity"]  = np.array(s.dq, np.float32)  # vitesses directes libfranka
                _latest["gripper_position"] = np.float32(gw)
        except Exception:
            pass
        time.sleep(0.01)    # 100 Hz suffisant pour 15 fps

# ── synchronizer — tick master 15 fps ────────────────────────────────────────
def _resize(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, (IMG_W, IMG_H))

def _synchronizer(stop: threading.Event, writers: dict,
                  frame_offset: int, task_index: int,
                  instruction: str) -> list[dict]:
    steps   = []
    t_next  = time.perf_counter()
    fi      = frame_offset
    warmup  = WARMUP_FRAMES

    while not stop.is_set():
        t_next += 1.0 / FPS

        with _lock:
            snap = {cam: _resize(
                        _latest[cam].copy() if _latest[cam] is not None
                        else np.zeros((RS_H, RS_W, 3), np.uint8))
                    for cam in CAMERAS}
            joint_pos  = _latest["joint_position"].copy()
            joint_vel  = _latest["joint_velocity"].copy()
            grip_pos   = float(_latest["gripper_position"])

        if warmup > 0:
            warmup -= 1   # tick horloge mais on jette la frame (cams encore vertes)
        else:
            for cam, frame in snap.items():
                writers[cam].write(frame)

            action = np.append(joint_vel, grip_pos)
            steps.append({
                "episode_index":     _total_episodes,
                "frame_index":       fi,
                "timestamp":         round((fi - frame_offset) / FPS, 6),
                "index":             fi,
                "task_index":        task_index,
                "task":              instruction,
                "joint_position":    joint_pos.tolist(),
                "gripper_position":  [grip_pos],
                "actions":           action.tolist(),
                "is_first":          fi == frame_offset,
                "is_last":           False,
                "is_terminal":       False,
            })
            fi += 1

        dt = t_next - time.perf_counter()
        if dt > 0:
            time.sleep(dt)

    if steps:
        steps[-1]["is_last"] = steps[-1]["is_terminal"] = True
    return steps

# ── save helpers ──────────────────────────────────────────────────────────────
def _append_parquet(path: Path, df_new: pd.DataFrame):
    if path.exists():
        df_new = pd.concat([pd.read_parquet(path), df_new], ignore_index=True)
    df_new.to_parquet(path, engine="pyarrow", index=False)

def _save_steps(steps: list[dict]):
    _append_parquet(_parquet_path(), pd.DataFrame(steps))

def _save_episode_meta(offset: int, length: int, task_index: int):
    _append_parquet(_episodes_path(), pd.DataFrame([{
        "episode_index": _total_episodes,
        "task_index":    task_index,
        "length":        length,
        "frame_offset":  offset,
        "file_index":    FILE_ID,
    }]))

def _save_meta():
    info = {
        "codebase_version": "v3.0",
        "robot_type":       "franka",
        "fps":              FPS,
        "total_episodes":   _total_episodes,
        "total_frames":     _total_frames,
        "total_tasks":      len(_tasks),
        "splits":           {"train": f"0:{_total_episodes}"},
        "features": {
            "wrist_image_left":      {"dtype": "video", "shape": [IMG_H, IMG_W, 3]},
            "exterior_image_1_left": {"dtype": "video", "shape": [IMG_H, IMG_W, 3]},
            "exterior_image_2_left": {"dtype": "video", "shape": [IMG_H, IMG_W, 3]},
            "joint_position":  {"dtype": "float32", "shape": [7],
                                "names": [f"joint_{i}" for i in range(7)]},
            "gripper_position": {"dtype": "float32", "shape": [1],
                                 "names": ["gripper"]},
            "actions":         {"dtype": "float32", "shape": [8],
                                "names": [f"joint_{i}_vel" for i in range(7)] + ["gripper"]},
        },
    }
    (DATA_ROOT / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    with open(DATA_ROOT / "meta" / "tasks.jsonl", "w") as f:
        for task, idx in _tasks.items():
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")

    p = _parquet_path()
    if p.exists():
        df = pd.read_parquet(p)
        def _s(col):
            mat = np.stack(df[col])
            return {"mean": mat.mean(0).tolist(), "std": mat.std(0).tolist(),
                    "min":  mat.min(0).tolist(),  "max": mat.max(0).tolist()}
        (DATA_ROOT / "meta" / "stats.json").write_text(
            json.dumps({"joint_position":  _s("joint_position"),
                        "gripper_position": _s("gripper_position"),
                        "actions":          _s("actions")}, indent=2))

# ── episode ───────────────────────────────────────────────────────────────────
def run_episode(writers: dict):
    global _total_frames, _total_episodes

    instruction = input("  Task instruction: ").strip() or "manipulation"
    t_idx       = _task_idx(instruction)
    frame_off   = _total_frames
    stop        = threading.Event()

    steps_box = [None]
    def _sync_worker():
        steps_box[0] = _synchronizer(stop, writers, frame_off, t_idx, instruction)

    threads = [
        threading.Thread(target=_realsense_thread, args=(stop,),                              daemon=True),
        threading.Thread(target=_arcam_thread,     args=(AR_DEVS[0], "exterior_image_1_left", stop), daemon=True),
        threading.Thread(target=_arcam_thread,     args=(AR_DEVS[1], "exterior_image_2_left", stop), daemon=True),
        threading.Thread(target=_franka_thread,    args=(stop,),                              daemon=True),
        threading.Thread(target=_sync_worker,                                                 daemon=True),
    ]
    for t in threads:
        t.start()

    print(f"\n● Episode {_total_episodes:06d}  \"{instruction}\"")
    input("  Press ENTER to stop...\n")
    stop.set()
    for t in threads:
        t.join(timeout=5)

    steps  = steps_box[0] or []
    length = len(steps)
    if length:
        t0 = time.time()
        _save_steps(steps)
        _save_episode_meta(frame_off, length, t_idx)
        _total_frames   += length
        _total_episodes += 1
        _save_meta()
        print(f"  Saved {length} frames in {time.time()-t0:.1f}s  →  ep {_total_episodes-1:06d}")
    else:
        print("  [WARN] Empty episode, skipped.")

# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    FILE_ID = _init_file_id()
    _make_dirs()

    writers = {cam: cv2.VideoWriter(
                   str(_vid_path(cam)), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (IMG_W, IMG_H))
               for cam in CAMERAS}

    print(f"=== Franka Recorder  file-{FILE_ID:03d} | {DATA_ROOT.resolve()} ===")
    print(f"    FPS={FPS}  résolution={IMG_W}×{IMG_H}  (capture RS:{RS_W}×{RS_H}, AR:{AR_W}×{AR_H})\n")

    def _shutdown(*_):
        print("\nClosing writers…")
        for w in writers.values():
            w.release()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)

    try:
        while True:
            input(f"\nPress ENTER to start episode {_total_episodes:06d}…")
            run_episode(writers)
    finally:
        for w in writers.values():
            w.release()