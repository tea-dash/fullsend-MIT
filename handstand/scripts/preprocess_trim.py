import argparse
import glob
import os
from typing import List

import ffmpeg  # type: ignore

from lib.io_utils import ensure_dir


def normalize_video(in_path: str, out_path: str, fps: int = 30, height: int = 720) -> None:
    (
        ffmpeg
        .input(in_path)
        .filter("fps", fps=fps)
        .filter("scale", -2, height)
        .output(out_path, an=None, vcodec="libx264", preset="fast", pix_fmt="yuv420p", r=fps)
        .overwrite_output()
        .run(quiet=True)
    )


def list_videos(path: str) -> List[str]:
    exts = ("*.mp4", "*.mov", "*.mkv", "*.avi")
    files: List[str] = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(path, ext)))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Normalize FPS/scale; export to trimmed dir (no auto-window here).")
    parser.add_argument("--split", choices=["pro", "user"], default="pro")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    raw_dir = os.path.join("data", args.split, "raw")
    out_dir = os.path.join("data", args.split, "trimmed")
    ensure_dir(out_dir)

    videos = list_videos(raw_dir)
    for vid in videos:
        base = os.path.splitext(os.path.basename(vid))[0]
        out_path = os.path.join(out_dir, f"{base}_30fps.mp4")
        print(f"Normalizing {vid} -> {out_path}")
        try:
            normalize_video(vid, out_path, fps=args.fps, height=args.height)
        except Exception as e:
            print(f"Failed to normalize {vid}: {e}")


if __name__ == "__main__":
    main()


