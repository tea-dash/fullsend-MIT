import argparse
import glob
import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from tqdm import tqdm

from lib.io_utils import ensure_dir, save_json, video_size_and_frames


def default_models():
    # Aliases from MMPose model zoo; adjust if needed
    det_model = "rtmdet_tiny_8xb32-300e_coco"
    pose_model = "td-hm_hrnet-w32_8xb64-210e_coco-256x192"
    return det_model, pose_model


def infer_video_mmpose(video_path: str, device: str = "cpu") -> List[Dict[str, Any]]:
    """
    Use MMPose high-level inferencer to extract 2D keypoints (COCO-17).
    Returns a list of per-frame dicts with fields: image_id, width, height, keypoints (17x3).
    """
    try:
        # Newer MMPose API
        from mmpose.apis import MMPoseInferencer  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "MMPose is required. Please install mmpose per requirements.txt. "
            f"Import error: {e}"
        )
    det_alias, pose_alias = default_models()
    inferencer = MMPoseInferencer(det_model=det_alias, pose2d=pose_alias, device=device)
    # results is a generator; each yield corresponds roughly to a frame result
    # We'll pre-load video sizes for metadata
    width, height, total_frames = video_size_and_frames(video_path)
    frames: List[Dict[str, Any]] = []
    idx = 0
    for result in inferencer(video_path, show=False, pred_out_dir=None):
        preds = result.get("predictions", None)
        # predictions is a list (per person) per frame; we choose the top score person
        best_keypoints: Optional[np.ndarray] = None
        best_score = -1.0
        if isinstance(preds, list) and len(preds) > 0:
            persons = preds[0] if isinstance(preds[0], list) else preds
            for person in persons:
                kpts = np.array(person.get("keypoints", []), dtype=float)  # (17,3)
                score = float(np.mean(kpts[:, 2])) if kpts.size else 0.0
                if kpts.size and score > best_score:
                    best_score = score
                    best_keypoints = kpts
        if best_keypoints is None:
            # no detection -> fill zeros
            best_keypoints = np.zeros((17, 3), dtype=float)
        frames.append(
            {
                "image_id": idx,
                "width": width,
                "height": height,
                "keypoints": best_keypoints.tolist(),
            }
        )
        idx += 1
    return frames


def process_split(split: str, device: str):
    in_dir = os.path.join("data", split, "trimmed")
    out_dir = os.path.join("data", split, "keypoints2d")
    ensure_dir(out_dir)
    videos = sorted(glob.glob(os.path.join(in_dir, "*.mp4")))
    for vid in tqdm(videos, desc=f"MMPose 2D ({split})"):
        base = os.path.splitext(os.path.basename(vid))[0]
        out_json = os.path.join(out_dir, f"{base}.json")
        if os.path.exists(out_json):
            continue
        try:
            frames = infer_video_mmpose(vid, device=device)
            save_json(frames, out_json)
        except Exception as e:
            print(f"Failed 2D inference for {vid}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run MMPose (detector+HRNet) to export 2D keypoints for a split.")
    parser.add_argument("--split", choices=["pro", "user"], default="pro")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()
    process_split(args.split, device=args.device)


if __name__ == "__main__":
    main()


