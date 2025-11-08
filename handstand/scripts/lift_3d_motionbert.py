import argparse
import glob
import os
import subprocess
from typing import Optional

from lib.io_utils import ensure_dir


def find_infer_script(motionbert_dir: str) -> Optional[str]:
    """
    Try a few likely paths for MotionBERT's infer_wild.py.
    """
    candidates = [
        os.path.join(motionbert_dir, "infer_wild.py"),
        os.path.join(motionbert_dir, "docs", "inference", "infer_wild.py"),
        os.path.join(motionbert_dir, "tools", "infer_wild.py"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def run_motionbert(infer_script: str, video_path: str, alphapose_json: str, out_dir: str) -> None:
    cmd = [
        "python",
        infer_script,
        "--vid_path",
        video_path,
        "--json_path",
        alphapose_json,
        "--out_path",
        out_dir,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=False)


def run_motionbert_lite(video_path: str, alphapose_json: str, out_dir: str, motionbert_dir: str) -> None:
    # Call our lite wrapper to avoid SMPL/vis deps
    cmd = [
        "python",
        "scripts/motionbert_inferlite.py",
        "--motionbert_dir",
        motionbert_dir,
        "--vid_path",
        video_path,
        "--json_path",
        alphapose_json,
        "--out_path",
        out_dir,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=False)


def process_split(split: str, motionbert_dir: str, use_lite: bool):
    infer_script = None
    if not use_lite:
        infer_script = find_infer_script(motionbert_dir)
        if not infer_script:
            raise RuntimeError(
                f"Could not find infer_wild.py in MOTIONBERT_DIR={motionbert_dir}. "
                "Please adjust the path or update lift_3d_motionbert.py."
            )
    key2d_dir = os.path.join("data", split, "keypoints2d")
    video_dir = os.path.join("data", split, "trimmed")
    out_dir = os.path.join("data", split, "poses3d")
    ensure_dir(out_dir)
    for ap_json in sorted(glob.glob(os.path.join(key2d_dir, "*.alphapose.json"))):
        base = os.path.splitext(os.path.basename(ap_json))[0].replace(".alphapose", "")
        vid_path = os.path.join(video_dir, f"{base}.mp4")
        if not os.path.exists(vid_path):
            # try alternative naming from preprocess
            alt = os.path.join(video_dir, f"{base}_30fps.mp4")
            if os.path.exists(alt):
                vid_path = alt
        sample_out_dir = os.path.join(out_dir, base)
        ensure_dir(sample_out_dir)
        if use_lite:
            run_motionbert_lite(vid_path, ap_json, sample_out_dir, motionbert_dir)
        else:
            run_motionbert(infer_script, vid_path, ap_json, sample_out_dir)


def main():
    parser = argparse.ArgumentParser(description="Lift 2D keypoints to 3D using MotionBERT infer_wild.py")
    parser.add_argument("--split", choices=["pro", "user"], default="pro")
    parser.add_argument(
        "--motionbert-dir",
        default=os.environ.get("MOTIONBERT_DIR", "models/motionbert_repo"),
        help="Path to MotionBERT repository (env MOTIONBERT_DIR overrides)",
    )
    parser.add_argument("--lite", action="store_true", help="Use lightweight wrapper (no SMPL/vis)")
    args = parser.parse_args()
    process_split(args.split, args.motionbert_dir, use_lite=args.lite)


if __name__ == "__main__":
    main()


