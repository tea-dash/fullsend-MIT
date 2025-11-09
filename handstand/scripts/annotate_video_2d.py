import argparse
import json
import os
import tempfile

import cv2
import numpy as np
import ffmpeg  # type: ignore

from lib.io_utils import ensure_dir
from lib.skeleton import COCO17_EDGES


def draw_kpts(image: np.ndarray, kpts: np.ndarray, color=(0, 255, 0)) -> np.ndarray:
    out = image.copy()
    for a, b in COCO17_EDGES:
        xa, ya, ca = kpts[a]
        xb, yb, cb = kpts[b]
        if ca > 0.05 and cb > 0.05:
            cv2.line(out, (int(xa), int(ya)), (int(xb), int(yb)), color, 2, cv2.LINE_AA)
    for x, y, c in kpts:
        if c > 0.05:
            cv2.circle(out, (int(x), int(y)), 3, (0, 128, 255), -1, cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser(description="Overlay 2D keypoints onto a video and save annotated output")
    parser.add_argument("--video", required=True, help="Path to video (normalized 30fps)")
    parser.add_argument("--keypoints_json", required=True, help="Per-frame JSON produced by extract_2d_*")
    parser.add_argument("--out", default=None, help="Output video path (mp4)")
    args = parser.parse_args()

    with open(args.keypoints_json, "r", encoding="utf-8") as f:
        frames = json.load(f)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {args.video}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out_path = args.out or os.path.splitext(args.video)[0] + "_annotated.mp4"
    # Write an intermediate MP4V and then transcode to H.264/yuv420p for browser compatibility
    tmp_path = os.path.splitext(out_path)[0] + "_mp4v.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps if fps > 0 else 30.0, (w, h))
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx < len(frames):
            kpts = np.array(frames[idx]["keypoints"], dtype=float)
            frame = draw_kpts(frame, kpts)
        writer.write(frame)
        idx += 1
    writer.release()
    cap.release()
    # Transcode to H.264 yuv420p + faststart
    try:
        (
            ffmpeg
            .input(tmp_path)
            .output(out_path, vcodec="libx264", pix_fmt="yuv420p", movflags="+faststart", r=fps if fps > 0 else 30.0, preset="fast")
            .overwrite_output()
            .run(quiet=True)
        )
        os.remove(tmp_path)
    except Exception as e:
        # Fallback to the MP4V file if ffmpeg not available
        if os.path.exists(out_path):
            os.remove(out_path)
        os.rename(tmp_path, out_path)
    print(f"Annotated video saved to {out_path}")


if __name__ == "__main__":
    main()


