"""
Replay the last (or any) episode from a LeRobot dataset in Rerun.

Usage:
    python replay.py                        # last episode
    python replay.py --episode 3            # episode #3
    python replay.py --dataset ./dataset    # custom dataset root
"""

import argparse
import json
import time
import numpy as np
import cv2
import pandas as pd
import rerun as rr
import rerun.blueprint as rrb
from pathlib import Path

# =========================
# CONFIG
# =========================

DATA_ROOT = Path("dataset")
CHUNK_ID  = 0
STATE_DIM = 8
ACTION_DIM = 7

CAMERAS = ["realsense", "cam_ar1", "cam_ar2", "tactile"]

# =========================
# DATASET DISCOVERY
# =========================

def find_file_id(root: Path) -> int:
    ref = root / "videos" / "observation.images.realsense" / "chunk-000"
    if not ref.exists():
        raise RuntimeError(f"No recording found in {root}")
    files = sorted(ref.glob("file-*.mp4"))
    if not files:
        raise RuntimeError(f"No video files found in {ref}")
    return len(files) - 1   # last file


def load_parquet(root: Path, file_id: int) -> pd.DataFrame:
    path = root / "data" / f"chunk-{CHUNK_ID:03d}" / f"file-{file_id:03d}.parquet"
    if not path.exists():
        raise RuntimeError(f"Parquet not found: {path}")
    return pd.read_parquet(path)


def load_episode_meta(root: Path, file_id: int) -> pd.DataFrame:
    path = (root / "meta" / "episodes"
            / f"chunk-{CHUNK_ID:03d}" / f"file-{file_id:03d}.parquet")
    if not path.exists():
        raise RuntimeError(f"Episodes meta not found: {path}")
    return pd.read_parquet(path)


def load_info(root: Path) -> dict:
    p = root / "meta" / "info.json"
    return json.loads(p.read_text()) if p.exists() else {}


def video_path(root: Path, cam: str, file_id: int) -> Path:
    return (root / "videos" / f"observation.images.{cam}"
            / f"chunk-{CHUNK_ID:03d}" / f"file-{file_id:03d}.mp4")

# =========================
# VIDEO EXTRACTION
# =========================

def open_video(path: Path, frame_offset: int) -> cv2.VideoCapture | None:
    if not path.exists():
        print(f"  [WARN] video missing: {path.name}")
        return None
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_offset)
    return cap


def read_frame(cap: cv2.VideoCapture | None) -> np.ndarray | None:
    if cap is None:
        return None
    ret, frame = cap.read()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# =========================
# BLUEPRINT
# =========================

def make_blueprint() -> rrb.Blueprint:
    """
    ┌──────────┬──────────┬──────────┬──────────┐
    │ realsense│ cam_ar1  │ cam_ar2  │  tactile │
    ├──────────┴──────────┴──────────┴──────────┤
    │              state / action tabs           │
    └────────────────────────────────────────────┘
    """
    cameras = rrb.Horizontal(
        rrb.Spatial2DView(name="RealSense", origin="/realsense"),
        rrb.Spatial2DView(name="ArCam 1",   origin="/cam_ar1"),
        rrb.Spatial2DView(name="ArCam 2",   origin="/cam_ar2"),
        rrb.Spatial2DView(name="Tactile",   origin="/tactile"),
    )
    state_tab = rrb.Horizontal(
        *[rrb.TimeSeriesView(name=f"joint_{i}", origin=f"/state/joint_{i}")
          for i in range(STATE_DIM)],
        name="State",
    )
    action_tab = rrb.Horizontal(
        *[rrb.TimeSeriesView(name=f"joint_{i}", origin=f"/action/joint_{i}")
          for i in range(ACTION_DIM)],
        name="Action",
    )
    return rrb.Blueprint(
        rrb.Vertical(
            cameras,
            rrb.Tabs(state_tab, action_tab),
            row_shares=[3, 2],
        ),
        rrb.BlueprintPanel(expanded=False),
        rrb.SelectionPanel(expanded=False),
        rrb.TimePanel(expanded=True),
    )

# =========================
# REPLAY
# =========================

def replay_episode(root: Path, episode_id: int | None, fps: int):
    file_id = find_file_id(root)
    df_all  = load_parquet(root, file_id)
    ep_meta = load_episode_meta(root, file_id)

    if episode_id is None:
        episode_id = int(ep_meta["episode_index"].max())

    row_ep = ep_meta[ep_meta["episode_index"] == episode_id].iloc[0]
    frame_offset = int(row_ep["frame_offset"])
    df = df_all[df_all["episode_index"] == episode_id].reset_index(drop=True)
    n  = len(df)

    print(f"  Episode      : {episode_id}  ({n} frames @ {fps} fps)")
    print(f"  Frame offset : {frame_offset}")

    caps = {cam: open_video(video_path(root, cam, file_id), frame_offset)
            for cam in CAMERAS}

    rr.init(f"replay · episode {episode_id:06d}", spawn=True)
    rr.send_blueprint(make_blueprint())

    print("\nStreaming at real-time speed...")

    for i, row in df.iterrows():
        t0 = time.perf_counter()

        rr.set_time("timestamp", timestamp=float(row["timestamp"]))
        rr.set_time("frame",     sequence=int(row["frame_index"]))

        # ─── Cameras ──────────────────────────────────────
        for cam in CAMERAS:
            img = read_frame(caps[cam])
            if img is not None:
                rr.log(f"/{cam}", rr.Image(img))

        # ─── State ────────────────────────────────────────
        state = np.array(row["observation.state"], dtype=np.float32)
        for j, val in enumerate(state):
            rr.log(f"/state/joint_{j}", rr.Scalars(float(val)))

        # ─── Action ───────────────────────────────────────
        action = np.array(row["action"], dtype=np.float32)
        for j, val in enumerate(action):
            rr.log(f"/action/joint_{j}", rr.Scalars(float(val)))

        # Sleep to log at real-time rate so Rerun plays at correct speed
        elapsed = time.perf_counter() - t0
        time.sleep(max(0.0, 1 / fps - elapsed))

        if (i + 1) % 30 == 0:
            print(f"  {100*(i+1)//n:3d}%  frame {i+1}/{n}", end="\r", flush=True)

    for cap in caps.values():
        if cap is not None:
            cap.release()

    print(f"\n✔ Done — episode {episode_id} ({n} frames)")

# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser(description="Replay a LeRobot episode in Rerun")
    parser.add_argument("--dataset", type=Path, default=DATA_ROOT)
    parser.add_argument("--episode", type=int, default=None)
    args = parser.parse_args()

    root = args.dataset
    if not root.exists():
        raise SystemExit(f"[ERROR] Dataset not found: {root.resolve()}")

    info = load_info(root)
    fps  = info.get("fps", 30)

    print(f"Dataset : {root.resolve()}")
    print(f"FPS     : {fps}")

    replay_episode(root, args.episode, fps)


if __name__ == "__main__":
    main()
