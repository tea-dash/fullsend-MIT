import argparse
import glob
import os
from typing import Any, Dict, List

import numpy as np

from lib.io_utils import ensure_dir, load_json, save_json
from lib.joints import COCO17_INDEX


def coco17_to_halpe26(kpts17: np.ndarray) -> np.ndarray:
    """
    Convert a (17,3) COCO array to a (26,3) Halpe array.
    Missing points (toes/heels) are approximated or zeroed.
    """
    out = np.zeros((26, 3), dtype=float)
    # Direct mappings
    out[0] = kpts17[COCO17_INDEX["nose"]]            # Nose
    out[1] = kpts17[COCO17_INDEX["left_eye"]]        # LEye
    out[2] = kpts17[COCO17_INDEX["right_eye"]]       # REye
    out[3] = kpts17[COCO17_INDEX["left_ear"]]        # LEar
    out[4] = kpts17[COCO17_INDEX["right_ear"]]       # REar
    out[5] = kpts17[COCO17_INDEX["left_shoulder"]]   # LShoulder
    out[6] = kpts17[COCO17_INDEX["right_shoulder"]]  # RShoulder
    out[7] = kpts17[COCO17_INDEX["left_elbow"]]      # LElbow
    out[8] = kpts17[COCO17_INDEX["right_elbow"]]     # RElbow
    out[9] = kpts17[COCO17_INDEX["left_wrist"]]      # LWrist
    out[10] = kpts17[COCO17_INDEX["right_wrist"]]    # RWrist
    out[11] = kpts17[COCO17_INDEX["left_hip"]]       # LHip
    out[12] = kpts17[COCO17_INDEX["right_hip"]]      # RHip
    out[13] = kpts17[COCO17_INDEX["left_knee"]]      # LKnee
    out[14] = kpts17[COCO17_INDEX["right_knee"]]     # Rknee
    out[15] = kpts17[COCO17_INDEX["left_ankle"]]     # LAnkle
    out[16] = kpts17[COCO17_INDEX["right_ankle"]]    # RAnkle
    # Head (17): approximate as a point above the nose along (nose - neck)
    nose = kpts17[COCO17_INDEX["nose"]]
    lsh = kpts17[COCO17_INDEX["left_shoulder"]]
    rsh = kpts17[COCO17_INDEX["right_shoulder"]]
    neck_xy = 0.5 * (lsh[:2] + rsh[:2])
    neck_c = min(lsh[2], rsh[2])
    dir_vec = nose[:2] - neck_xy
    if np.linalg.norm(dir_vec) < 1e-6:
        head_xy = nose[:2]
        head_c = nose[2]
    else:
        # One head-length above nose
        head_xy = nose[:2] + dir_vec
        head_c = min(float(nose[2]), float(neck_c))
    out[17] = np.array([head_xy[0], head_xy[1], head_c], dtype=float)
    # Neck (18): midpoint of shoulders
    # Neck (18): midpoint of shoulders
    out[18] = np.array([neck_xy[0], neck_xy[1], neck_c], dtype=float)
    # Hip center (19): midpoint of hips
    lhip = kpts17[COCO17_INDEX["left_hip"]]
    rhip = kpts17[COCO17_INDEX["right_hip"]]
    hip_xy = 0.5 * (lhip[:2] + rhip[:2])
    hip_c = min(lhip[2], rhip[2])
    out[19] = np.array([hip_xy[0], hip_xy[1], hip_c], dtype=float)
    # Toes/heels (20-25): leave zeros (unused in halpe2h36m)
    return out


def process_split(split: str):
    in_dir = os.path.join("data", split, "keypoints2d")
    out_dir = in_dir
    ensure_dir(out_dir)
    # Only process raw per-frame JSONs (exclude already-converted AlphaPose files)
    candidates = [p for p in sorted(glob.glob(os.path.join(in_dir, "*.json"))) if "alphapose" not in os.path.basename(p)]
    for in_json in candidates:
        base = os.path.splitext(os.path.basename(in_json))[0]
        out_json = os.path.join(out_dir, f"{base}.alphapose.json")
        frames: List[Dict[str, Any]] = load_json(in_json)
        # If this file already looks like AlphaPose (keys per record with 26*3), skip
        if isinstance(frames, list) and frames:
            sample = frames[0]
            if "category_id" in sample or (isinstance(sample, dict) and "keypoints" in sample and len(sample["keypoints"]) in (26*3, 17*3) and "width" not in sample):
                # Appears to be AlphaPose-like already; write-through or skip
                save_json(frames, out_json)
                print(f"Detected AlphaPose-like input; wrote through: {out_json}")
                continue
        recs: List[Dict[str, Any]] = []
        for f in frames:
            k17 = np.array(f["keypoints"], dtype=float).reshape(17, 3)
            k26 = coco17_to_halpe26(k17)
            score = float(np.mean(k26[:, 2]))
            recs.append(
                {
                    "image_id": f.get("image_id", 0),
                    "category_id": 1,
                    "keypoints": k26.reshape(-1).tolist(),
                    "score": score,
                }
            )
        save_json(recs, out_json)
        print(f"Wrote Halpe26 AlphaPose JSON: {out_json}")


def main():
    parser = argparse.ArgumentParser(description="Convert COCO-17 per-frame JSON to Halpe-26 AlphaPose JSON")
    parser.add_argument("--split", choices=["pro", "user"], default="user")
    args = parser.parse_args()
    process_split(args.split)


if __name__ == "__main__":
    main()


