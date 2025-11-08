import argparse
import glob
import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from tqdm import tqdm

from lib.io_utils import ensure_dir, save_json, video_size_and_frames


def infer_video_ultra(video_path: str, device: str = "cpu") -> List[Dict[str, Any]]:
    """
    Use Ultralytics YOLOv8n-pose to extract 2D keypoints (COCO-17).
    Returns a list of per-frame dicts: image_id, width, height, keypoints (17x3).
    """
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Ultralytics is required. Please install via `pip install ultralytics`."
        ) from e
    model = YOLO("yolov8n-pose.pt")
    if device != "cpu":
        model.to(device)
    width, height, total_frames = video_size_and_frames(video_path)
    results = model(video_path, stream=True, verbose=False)
    frames: List[Dict[str, Any]] = []
    idx = 0
    for r in results:
        best_kpts: Optional[np.ndarray] = None
        best_score = -1.0
        if r.keypoints is not None and len(r.keypoints) > 0:
            # r.keypoints.data: (N, 17, 3) with (x,y,conf)
            kpts_all = r.keypoints.data.cpu().numpy()
            confs = kpts_all[..., 2].mean(axis=1)
            i = int(np.argmax(confs))
            best_kpts = kpts_all[i]
            best_score = float(confs[i])
        if best_kpts is None:
            best_kpts = np.zeros((17, 3), dtype=float)
        frames.append(
            {
                "image_id": idx,
                "width": width,
                "height": height,
                "keypoints": best_kpts.tolist(),
            }
        )
        idx += 1
    return frames


def process_split(split: str, device: str):
    in_dir = os.path.join("data", split, "trimmed")
    out_dir = os.path.join("data", split, "keypoints2d")
    ensure_dir(out_dir)
    videos = sorted(glob.glob(os.path.join(in_dir, "*.mp4")))
    for vid in tqdm(videos, desc=f"Ultralytics 2D ({split})"):
        base = os.path.splitext(os.path.basename(vid))[0]
        out_json = os.path.join(out_dir, f"{base}.json")
        if os.path.exists(out_json):
            continue
        try:
            frames = infer_video_ultra(vid, device=device)
            save_json(frames, out_json)
        except Exception as e:
            print(f"Failed ultralytics 2D for {vid}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run Ultralytics YOLOv8 pose to export 2D keypoints.")
    parser.add_argument("--split", choices=["pro", "user"], default="pro")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()
    process_split(args.split, device=args.device)


if __name__ == "__main__":
    main()


