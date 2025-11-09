import argparse
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np  # noqa: E402

from lib.io_utils import load_npz, ensure_dir
from lib.skeleton import H36M_EDGES
from lib.joints import H36M_17_NAMES


def set_equal_3d_axes(ax, xs, ys, zs, pad_ratio: float = 0.15):
    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    y_min, y_max = float(np.min(ys)), float(np.max(ys))
    z_min, z_max = float(np.min(zs)), float(np.max(zs))
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    cz = 0.5 * (z_min + z_max)
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    if max_range <= 0:
        max_range = 1.0
    pad = pad_ratio * max_range
    half = 0.5 * max_range + pad
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_zlim(cz - half, cz + half)


def render_debug_video(aligned: np.ndarray, out_mp4: str, fps: int = 30):
    T, J, _ = aligned.shape
    ensure_dir(os.path.dirname(out_mp4) or ".")

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    xs = aligned[:, :, 0].ravel()
    ys = aligned[:, :, 1].ravel()
    zs = aligned[:, :, 2].ravel()
    set_equal_3d_axes(ax, xs, ys, zs)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Axis length based on scene scale
    span = max(xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min())
    if not np.isfinite(span) or span <= 0:
        span = 1.0
    axis_len = 0.5 * span

    # Plot handles
    scat = ax.scatter([], [], [], c="orange", s=25, depthshade=True)
    lines = [ax.plot([], [], [], c="blue", linewidth=2)[0] for _ in H36M_EDGES]
    ax_x = ax.plot([], [], [], c="red", linewidth=2)[0]
    ax_y = ax.plot([], [], [], c="green", linewidth=2)[0]
    ax_z = ax.plot([], [], [], c="royalblue", linewidth=2)[0]
    # Per-joint label artists
    labels = [ax.text(0, 0, 0, "", fontsize=8, color="black") for _ in range(J)]
    txt = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    def init():
        scat._offsets3d = ([], [], [])
        for ln in lines:
            ln.set_data([], [])
            ln.set_3d_properties([])
        for a in (ax_x, ax_y, ax_z):
            a.set_data([], [])
            a.set_3d_properties([])
        for t in labels:
            t.set_text("")
        txt.set_text("")
        return [scat, *lines, ax_x, ax_y, ax_z, *labels, txt]

    def update(t):
        pts = aligned[t]  # (J,3)
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        scat._offsets3d = (x, y, z)
        for (a, b), ln in zip(H36M_EDGES, lines):
            ln.set_data([x[a], x[b]], [y[a], y[b]])
            ln.set_3d_properties([z[a], z[b]])
        # Axes
        ox = oy = oz = 0.0
        ax_x.set_data([ox, ox + axis_len], [oy, oy]); ax_x.set_3d_properties([oz, oz])
        ax_y.set_data([ox, ox], [oy, oy + axis_len]); ax_y.set_3d_properties([oz, oz])
        ax_z.set_data([ox, ox], [oy, oy]); ax_z.set_3d_properties([oz, oz + axis_len])
        # Labels with index:name
        for i, lab in enumerate(labels):
            lab.set_position((x[i], y[i]))
            lab.set_3d_properties(z[i])
            lab.set_text(f"{i}:{H36M_17_NAMES[i]}")
        txt.set_text(f"Frame {t+1}/{T}")
        return [scat, *lines, ax_x, ax_y, ax_z, *labels, txt]

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        plt.rcParams["animation.ffmpeg_path"] = ffmpeg_path
    writer = animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=4000,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    ani = animation.FuncAnimation(fig, update, frames=T, init_func=init, blit=False, interval=1000.0 / max(fps, 1))
    ani.save(out_mp4, writer=writer)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Render 3D debug video with H36M joint labels and axes")
    parser.add_argument("--aligned_npz", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    data = load_npz(args.aligned_npz)
    aligned = data["aligned"]
    out = args.out or os.path.splitext(args.aligned_npz)[0] + "_3d_debug.mp4"
    render_debug_video(aligned, out, fps=args.fps)
    print(f"Saved 3D debug video to {out}")


if __name__ == "__main__":
    main()


