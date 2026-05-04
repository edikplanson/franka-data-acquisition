"""
run.py — Replay last episode with Rerun.io

Usage:
    python run.py                        # dernier épisode
    python run.py --episode 3            # épisode #3
    python run.py --dataset ./dataset    # autre dossier
"""

import argparse, json, time
import numpy as np
import cv2
import pandas as pd
import rerun as rr
import rerun.blueprint as rrb
from pathlib import Path

DATA_ROOT = Path("dataset")
CHUNK_ID  = 0
CAMERAS   = ["wrist_image_left", "exterior_image_1_left", "exterior_image_2_left"]
N_JOINTS  = 7

# ── dataset helpers ───────────────────────────────────────────────────────────
def _all_episode_files(root: Path) -> list[Path]:
    """All episode-meta parquet files, sorted by modification time (oldest first)."""
    meta_dir = root / "meta" / "episodes" / f"chunk-{CHUNK_ID:03d}"
    if not meta_dir.exists():
        raise RuntimeError(f"No episode metadata in {root}")
    files = list(meta_dir.glob("file-*.parquet"))
    if not files:
        raise RuntimeError("No episode metadata files found — record at least one episode first.")
    return sorted(files, key=lambda p: p.stat().st_mtime)

def _find_episode(root: Path, episode_id: int | None) -> tuple[int, int]:
    """
    Returns (file_id, local_episode_index).
    If episode_id is None  → last recorded episode (most-recent file, highest index).
    If episode_id is given → search globally across all files in recording order.
    """
    files = _all_episode_files(root)

    if episode_id is None:
        last = files[-1]
        meta = pd.read_parquet(last)
        file_id = int(last.stem.split("-")[1])
        return file_id, int(meta["episode_index"].max())

    # global sequential lookup: files sorted by mtime, episodes sorted by index within each
    global_idx = 0
    for f in files:
        meta    = pd.read_parquet(f)
        indices = sorted(meta["episode_index"].unique())
        for local_idx in indices:
            if global_idx == episode_id:
                return int(f.stem.split("-")[1]), local_idx
            global_idx += 1

    raise RuntimeError(f"Episode {episode_id} not found (dataset has {global_idx} episodes total).")

def _load(root: Path, file_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_parquet(
        root / "data" / f"chunk-{CHUNK_ID:03d}" / f"file-{file_id:03d}.parquet")
    meta = pd.read_parquet(
        root / "meta" / "episodes" / f"chunk-{CHUNK_ID:03d}" / f"file-{file_id:03d}.parquet")
    return data, meta

def _vid(root: Path, cam: str, file_id: int) -> Path:
    return root / "videos" / cam / f"chunk-{CHUNK_ID:03d}" / f"file-{file_id:03d}.mp4"

# ── rerun blueprint ───────────────────────────────────────────────────────────
def _blueprint() -> rrb.Blueprint:
    cams = rrb.Horizontal(
        *[rrb.Spatial2DView(name=c.replace("_", " ").title(), origin=f"/{c}")
          for c in CAMERAS],
    )
    joints = rrb.Horizontal(
        *[rrb.TimeSeriesView(name=f"j{i}", origin=f"/state/j{i}") for i in range(N_JOINTS)],
    )
    actions = rrb.Horizontal(
        *[rrb.TimeSeriesView(name=f"dq{i}", origin=f"/action/dq{i}") for i in range(N_JOINTS)],
    )
    return rrb.Blueprint(
        rrb.Vertical(
            cams,
            rrb.Tabs(
                rrb.Horizontal(joints,  name="Positions"),
                rrb.Horizontal(actions, name="Vitesses (dq)"),
                rrb.TextDocumentView(name="Task", origin="/task"),
            ),
            row_shares=[3, 2],
        ),
        rrb.BlueprintPanel(expanded=False),
        rrb.SelectionPanel(expanded=False),
        rrb.TimePanel(expanded=True),
    )

# ── replay ────────────────────────────────────────────────────────────────────
def replay(root: Path, episode_id: int | None, fps: int):
    file_id, local_ep = _find_episode(root, episode_id)
    df_all, meta      = _load(root, file_id)
    episode_id        = local_ep

    ep_row = meta[meta["episode_index"] == episode_id].iloc[0]
    offset = int(ep_row["frame_offset"])
    df     = df_all[df_all["episode_index"] == episode_id].reset_index(drop=True)
    n      = len(df)

    instruction = str(df.iloc[0].get("task", "")) if n else ""

    caps: dict[str, cv2.VideoCapture | None] = {}
    for cam in CAMERAS:
        p = _vid(root, cam, file_id)
        if p.exists():
            cap = cv2.VideoCapture(str(p))
            cap.set(cv2.CAP_PROP_POS_FRAMES, offset)
            caps[cam] = cap
        else:
            print(f"  [WARN] missing video: {p.name}")
            caps[cam] = None

    rr.init(f"ep{episode_id:06d} — {instruction}", spawn=True)
    rr.send_blueprint(_blueprint())

    print(f"\nEpisode  : {episode_id}")
    print(f"Frames   : {n} @ {fps} fps")
    print(f"Task     : {instruction}\n")

    for i, row in df.iterrows():
        t0 = time.perf_counter()

        rr.set_time("timestamp", timestamp=float(row["timestamp"]))
        rr.set_time("frame",     sequence=int(row["frame_index"]))

        for cam, cap in caps.items():
            if cap is not None:
                ok, frame = cap.read()
                if ok:
                    rr.log(f"/{cam}", rr.Image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

        joint_pos = np.array(row["joint_position"], dtype=np.float32)
        for j, v in enumerate(joint_pos):
            rr.log(f"/state/j{j}", rr.Scalars(float(v)))

        actions = np.array(row["actions"], dtype=np.float32)
        for j, v in enumerate(actions[:N_JOINTS]):
            rr.log(f"/action/dq{j}", rr.Scalars(float(v)))

        rr.log("/task", rr.TextDocument(instruction))

        dt = 1.0 / fps - (time.perf_counter() - t0)
        if dt > 0:
            time.sleep(dt)

        if (i + 1) % fps == 0:
            print(f"  {100*(i+1)//n:3d}%  frame {i+1}/{n}", end="\r", flush=True)

    for cap in caps.values():
        if cap:
            cap.release()

    print(f"\nDone — episode {episode_id:06d}  ({n} frames)")

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Replay a LeRobot episode in Rerun")
    ap.add_argument("--dataset", type=Path, default=DATA_ROOT)
    ap.add_argument("--episode", type=int,  default=None)
    args = ap.parse_args()

    if not args.dataset.exists():
        raise SystemExit(f"[ERROR] Dataset not found: {args.dataset.resolve()}")

    info_path = args.dataset / "meta" / "info.json"
    fps = json.loads(info_path.read_text()).get("fps", 30) if info_path.exists() else 30

    replay(args.dataset, args.episode, fps)


if __name__ == "__main__":
    main()
