import argparse
import glob
import os
from typing import Any, Dict, List

import numpy as np

from lib.io_utils import ensure_dir, load_json, save_json


def to_alphapose_records(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert our per-frame JSON (one person) into a list of AlphaPose-like records.
    Each record includes 'image_id', 'category_id', 'keypoints' (flat 51-length list), and 'score'.
    """
    records: List[Dict[str, Any]] = []
    for f in frames:
        kpts = np.array(f["keypoints"], dtype=float)  # (17,3)
        score = float(np.mean(kpts[:, 2])) if kpts.size else 0.0
        flat = kpts.reshape(-1).tolist()
        records.append(
            {
                "image_id": f["image_id"],
                "category_id": 1,
                "keypoints": flat,
                "score": score,
            }
        )
    return records


def process_split(split: str):
    in_dir = os.path.join("data", split, "keypoints2d")
    out_dir = os.path.join("data", split, "keypoints2d")
    ensure_dir(out_dir)
    json_files = sorted(glob.glob(os.path.join(in_dir, "*.json")))
    for jf in json_files:
        base = os.path.splitext(os.path.basename(jf))[0]
        out_json = os.path.join(out_dir, f"{base}.alphapose.json")
        if os.path.exists(out_json):
            continue
        frames = load_json(jf)
        recs = to_alphapose_records(frames)
        save_json(recs, out_json)


def main():
    parser = argparse.ArgumentParser(description="Convert MMPose per-frame JSON to AlphaPose-like JSON.")
    parser.add_argument("--split", choices=["pro", "user"], default="pro")
    args = parser.parse_args()
    process_split(args.split)


if __name__ == "__main__":
    main()


