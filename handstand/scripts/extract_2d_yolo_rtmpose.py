import argparse
import glob
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from lib.io_utils import ensure_dir, save_json, video_size_and_frames
import sys

# Prepend local vendor stubs (xtcocotools -> pycocotools) before importing mmpose
VENDOR_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vendor")
if VENDOR_PATH not in sys.path:
    sys.path.insert(0, VENDOR_PATH)


def yolo_person_bbox(model, frame: np.ndarray) -> Optional[Tuple[float, float, float, float, float]]:
    """
    Run YOLOv8 detector and return the largest person bbox (x1,y1,x2,y2,score).
    """
    res = model.predict(frame, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None
    boxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    clss = res.boxes.cls.cpu().numpy().astype(int)
    best = None
    best_metric = -1.0
    for (x1, y1, x2, y2), conf, cls in zip(boxes, confs, clss):
        if cls != 0:  # person class
            continue
        area = (x2 - x1) * (y2 - y1)
        metric = area * float(conf)
        if metric > best_metric:
            best_metric = metric
            best = (float(x1), float(y1), float(x2), float(y2), float(conf))
    return best


def infer_crop_rtmpose(inferencer, crop: np.ndarray) -> Optional[np.ndarray]:
    """
    Run MMPose RTMPose on a cropped person image and return (17,3) keypoints in crop coords.
    """
    # MMPoseInferencer returns a generator of results dicts
    try:
        gen = inferencer(crop, return_datasamples=False, show=False)
    except Exception as e:
        print(f"MMPose error: {e}")
        return None
    result = next(gen, None)
    if result is None:
        return None
    preds = result.get("predictions", None)
    if preds is None:
        return None
    # preds is usually a list with one element (current input)
    persons = preds[0] if isinstance(preds[0], list) else preds
    if not persons:
        return None
    # Choose highest score person
    best = None
    best_score = -1.0
    for p in persons:
        kpts = np.array(p.get("keypoints", []), dtype=float)
        if kpts.size == 0:
            continue
        score = float(np.mean(kpts[:, 2]))
        if score > best_score:
            best_score = score
            best = kpts
    return best


def process_video_yolo_rtmpose(video_path: str, out_json: str, device: str = "cpu"):
    from ultralytics import YOLO  # type: ignore
    from mmpose.apis import MMPoseInferencer  # type: ignore

    # Detector: YOLOv8x for stronger person boxes
    det = YOLO("yolov8x.pt")
    if device != "cpu":
        det.to(device)
    # Pose model: largest RTMPose on COCO, 384x288
    pose_alias = "rtmpose-x_8xb64-700e_coco-384x288"
    pose = MMPoseInferencer(pose2d=pose_alias, device=device, det_model=None)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames: List[Dict[str, Any]] = []
    idx = 0
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0, desc="YOLO+RTMPose")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        bbox = yolo_person_bbox(det, frame)
        if bbox is None:
            kpts = np.zeros((17, 3), dtype=float)
        else:
            x1, y1, x2, y2, score = bbox
            # Pad bbox slightly
            pad = 0.05
            dx = (x2 - x1) * pad
            dy = (y2 - y1) * pad
            xx1 = max(0, int(x1 - dx))
            yy1 = max(0, int(y1 - dy))
            xx2 = min(width - 1, int(x2 + dx))
            yy2 = min(height - 1, int(y2 + dy))
            crop = frame[yy1:yy2, xx1:xx2]
            kpts_crop = infer_crop_rtmpose(pose, crop)
            if kpts_crop is None:
                kpts = np.zeros((17, 3), dtype=float)
            else:
                # Map keypoints back to original frame coordinates
                kpts = kpts_crop.copy()
                kpts[:, 0] += xx1
                kpts[:, 1] += yy1
        frames.append(
            {
                "image_id": idx,
                "width": width,
                "height": height,
                "keypoints": kpts.tolist(),
            }
        )
        idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    ensure_dir(os.path.dirname(out_json))
    save_json(frames, out_json)


def process_split(split: str, device: str):
    in_dir = os.path.join("data", split, "trimmed")
    out_dir = os.path.join("data", split, "keypoints2d")
    ensure_dir(out_dir)
    for vid in sorted(glob.glob(os.path.join(in_dir, "*.mp4"))):
        base = os.path.splitext(os.path.basename(vid))[0]
        out_json = os.path.join(out_dir, f"{base}.json")
        if os.path.exists(out_json):
            continue
        process_video_yolo_rtmpose(vid, out_json, device=device)


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 (det) + MMPose RTMPose-X 2D extraction")
    parser.add_argument("--split", choices=["pro", "user"], default="user")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()
    process_split(args.split, device=args.device)


if __name__ == "__main__":
    main()


