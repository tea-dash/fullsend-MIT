import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np  # noqa: E402

from lib.io_utils import load_npz, ensure_dir
from lib.skeleton import H36M_EDGES


def render_frame_3d(points: np.ndarray, out_png: str) -> None:
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
    ax.scatter(xs, ys, zs, c="orange", s=20, depthshade=True)
    for a, b in H36M_EDGES:
        ax.plot([xs[a], xs[b]], [ys[a], ys[b]], [zs[a], zs[b]], c="blue", linewidth=2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # Set equal aspect
    max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max()
    Xb = 0.5 * max_range
    mid_x, mid_y, mid_z = xs.mean(), ys.mean(), zs.mean()
    ax.set_xlim(mid_x - Xb, mid_x + Xb)
    ax.set_ylim(mid_y - Xb, mid_y + Xb)
    ax.set_zlim(mid_z - Xb, mid_z + Xb)
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Render a 3D keypoint image from aligned 3D npz")
    parser.add_argument("--aligned_npz", required=True)
    parser.add_argument("--frame", type=int, default=None, help="Frame index; default: mid frame")
    parser.add_argument("--out", default=None, help="Output PNG path")
    args = parser.parse_args()
    data = load_npz(args.aligned_npz)
    aligned = data["aligned"]
    T = aligned.shape[0]
    frame_idx = args.frame if args.frame is not None else (T // 2)
    frame_idx = max(0, min(T - 1, frame_idx))
    out = args.out or os.path.splitext(args.aligned_npz)[0] + f"_3d_f{frame_idx}.png"
    ensure_dir(os.path.dirname(out) or ".")
    render_frame_3d(aligned[frame_idx], out)
    print(f"Saved 3D image to {out}")


if __name__ == "__main__":
    main()


