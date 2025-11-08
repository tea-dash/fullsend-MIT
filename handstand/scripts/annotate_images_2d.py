import argparse
import glob
import os
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO  # type: ignore

from lib.io_utils import ensure_dir
from lib.joints import COCO17_INDEX
from lib.skeleton import COCO17_EDGES


def draw_kpts(image: np.ndarray, kpts: np.ndarray, color=(0, 255, 0)) -> np.ndarray:
    out = image.copy()
    # draw lines
    for a, b in COCO17_EDGES:
        xa, ya, ca = kpts[a]
        xb, yb, cb = kpts[b]
        if ca > 0.05 and cb > 0.05:
            cv2.line(out, (int(xa), int(ya)), (int(xb), int(yb)), color, 2, cv2.LINE_AA)
    # draw points
    for x, y, c in kpts:
        if c > 0.05:
            cv2.circle(out, (int(x), int(y)), 3, (0, 128, 255), -1, cv2.LINE_AA)
    return out


def infer_image(model: YOLO, path: str) -> np.ndarray:
    res = model(path, verbose=False)[0]
    if res.keypoints is None or len(res.keypoints) == 0:
        h, w = cv2.imread(path).shape[:2]
        return np.zeros((17, 3), dtype=float)
    kpts_all = res.keypoints.data.cpu().numpy()  # (N,17,3)
    confs = kpts_all[..., 2].mean(axis=1)
    i = int(np.argmax(confs))
    return kpts_all[i]


def main():
    parser = argparse.ArgumentParser(description="Annotate 2D keypoints on images using YOLOv8-pose")
    parser.add_argument("--in-dir", default="data/user/raw")
    parser.add_argument("--out-dir", default="data/user/annotated_images")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    model = YOLO("yolov8n-pose.pt")
    if args.device != "cpu":
        model.to(args.device)

    exts = ("*.jpg", "*.jpeg", "*.png")
    files: List[str] = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(args.in_dir, ext)))
    for path in files:
        img = cv2.imread(path)
        if img is None:
            continue
        kpts = infer_image(model, path)
        vis = draw_kpts(img, kpts)
        base = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(os.path.join(args.out_dir, f"{base}_2d.jpg"), vis)
        print(f"Saved {base}_2d.jpg")


if __name__ == "__main__":
    main()


