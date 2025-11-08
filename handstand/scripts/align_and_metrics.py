import argparse
import glob
import json
import os
from typing import Dict, Optional, Tuple

import numpy as np

from lib.io_utils import ensure_dir, load_npz, save_json, save_npz, video_size_and_frames
from lib.joints import COCO17_NAMES
from lib.metrics import align_sequence, summarize_sequence_metrics


def find_3d_sequence(sample_dir: str) -> Optional[Tuple[np.ndarray, str]]:
    """
    Locate a 3D pose file in a MotionBERT output directory and return (T,J,3) and a source path.
    """
    # Try NPZ first
    for npz_path in glob.glob(os.path.join(sample_dir, "*.npz")):
        data = load_npz(npz_path)
        for key in ["preds", "pred_3d", "poses_3d", "joint3d"]:
            if key in data:
                arr = np.array(data[key])
                if arr.ndim == 3 and arr.shape[2] == 3:
                    return arr.astype(float), npz_path
            # sometimes shape could be (J,3,T)
            if key in data:
                arr = np.array(data[key])
                if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[2] == len(COCO17_NAMES):
                    arr = np.transpose(arr, (2, 1, 0))
                    return arr.astype(float), npz_path
    # Try JSON
    for js_path in glob.glob(os.path.join(sample_dir, "*.json")):
        try:
            with open(js_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            # Expect list of frames with 'pose_3d' or similar
            if isinstance(obj, list) and obj and "pose_3d" in obj[0]:
                seq = np.array([frame["pose_3d"] for frame in obj], dtype=float)  # T x (J*3)
                T, flat = seq.shape
                J = flat // 3
                seq = seq.reshape(T, J, 3)
                return seq, js_path
        except Exception:
            continue
    # Try standard MotionBERT output NPY
    npy_path = os.path.join(sample_dir, "X3D.npy")
    if os.path.exists(npy_path):
        try:
            arr = np.load(npy_path)
            # Expected shape (T, J, 3)
            if arr.ndim == 3 and arr.shape[2] == 3:
                return arr.astype(float), npy_path
            # Sometimes (N*T, J, 3) or (J, 3, T) – handle common transpose
            if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[1] == len(COCO17_NAMES):
                arr = np.transpose(arr, (2, 1, 0))
                return arr.astype(float), npy_path
        except Exception:
            pass
    return None


def process_split(split: str):
    poses_root = os.path.join("data", split, "poses3d")
    aligned_root = os.path.join("data", split, "aligned")
    metrics_root = os.path.join("data", split, "metrics")
    ensure_dir(aligned_root)
    ensure_dir(metrics_root)

    # Determine FPS via corresponding video when possible
    video_root = os.path.join("data", split, "trimmed")

    for sample_dir in sorted(glob.glob(os.path.join(poses_root, "*"))):
        if not os.path.isdir(sample_dir):
            continue
        base = os.path.basename(sample_dir)
        out_aligned_npz = os.path.join(aligned_root, f"{base}.npz")
        out_metrics_json = os.path.join(metrics_root, f"{base}.json")
        if os.path.exists(out_metrics_json) and os.path.exists(out_aligned_npz):
            continue
        result = find_3d_sequence(sample_dir)
        if not result:
            print(f"Skipping {sample_dir}: no recognizable 3D output found")
            continue
        seq3d, source_path = result
        # FPS from video if exists
        vid_path = os.path.join(video_root, f"{base}.mp4")
        if not os.path.exists(vid_path):
            alt = os.path.join(video_root, f"{base}_30fps.mp4")
            vid_path = alt if os.path.exists(alt) else None  # type: ignore
        fps = 30.0
        if vid_path:
            try:
                w, h, frames = video_size_and_frames(vid_path)
                # OpenCV CAP_PROP_FPS is not always reliable; assume 30 due to preprocessing
                fps = 30.0
            except Exception:
                pass
        aligned = align_sequence(seq3d)
        metrics = summarize_sequence_metrics(aligned, fps=fps)
        save_npz(out_aligned_npz, aligned=aligned, joint_names=np.array(COCO17_NAMES))
        save_json(metrics, out_metrics_json)
        print(f"Aligned+metrics saved for {base}")


def main():
    parser = argparse.ArgumentParser(description="Align 3D poses and compute metrics for a split.")
    parser.add_argument("--split", choices=["pro", "user"], default="pro")
    args = parser.parse_args()
    process_split(args.split)


if __name__ == "__main__":
    main()


