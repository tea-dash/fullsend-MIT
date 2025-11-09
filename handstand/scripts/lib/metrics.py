from typing import Dict, List, Tuple

import numpy as np

from .geometry import (
    canonical_axes,
    transform_points,
    angle_three_points,
    vector_angle,
    mid_point,
    normalize_vector,
)
from .joints import H36M_INDEX


def compute_body_axes_frame(points_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute origin and rotation axes from a single frame using hips and neck.
    points_xyz: (J, 3), H36M-17 order.
    """
    left_hip = points_xyz[H36M_INDEX["left_hip"]]
    right_hip = points_xyz[H36M_INDEX["right_hip"]]
    neck = points_xyz[H36M_INDEX["neck"]]
    origin, R = canonical_axes(left_hip, right_hip, neck)
    # Ensure Y axis derives from pelvis-to-neck direction, keep orthonormal basis
    pelvis = points_xyz[H36M_INDEX["pelvis"]]
    z_axis = normalize_vector(neck - pelvis)
    x_axis = normalize_vector(right_hip - left_hip)
    y_axis = normalize_vector(np.cross(z_axis, x_axis))
    x_axis = normalize_vector(np.cross(y_axis, z_axis))
    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    return pelvis, R


def align_sequence(points_xyz_seq: np.ndarray) -> np.ndarray:
    """
    Align a sequence of 3D joints using per-frame body-centric coordinates.
    points_xyz_seq: (T, J, 3)
    """
    T = points_xyz_seq.shape[0]
    aligned = np.zeros_like(points_xyz_seq)
    for t in range(T):
        origin, R = compute_body_axes_frame(points_xyz_seq[t])
        aligned_frame = transform_points(points_xyz_seq[t], origin, R)
        pelvis = aligned_frame[H36M_INDEX["pelvis"]]
        aligned_frame -= pelvis
        aligned_frame[:, 2] *= -1  # flip Z so head points +Z
        aligned[t] = aligned_frame
    return aligned


def joint_angle_triplet(points_xyz: np.ndarray, a: str, b: str, c: str) -> float:
    ai = H36M_INDEX[a]
    bi = H36M_INDEX[b]
    ci = H36M_INDEX[c]
    return angle_three_points(points_xyz[ai], points_xyz[bi], points_xyz[ci])


def compute_core_angles(points_xyz: np.ndarray) -> Dict[str, float]:
    """
    Compute key joint angles (degrees) for a single frame in aligned space.
    """
    angles: Dict[str, float] = {}
    # Elbows
    angles["left_elbow"] = joint_angle_triplet(points_xyz, "left_shoulder", "left_elbow", "left_wrist")
    angles["right_elbow"] = joint_angle_triplet(points_xyz, "right_shoulder", "right_elbow", "right_wrist")
    # Shoulders relative to neck
    neck = points_xyz[H36M_INDEX["neck"]]
    angles["left_shoulder"] = angle_three_points(neck, points_xyz[H36M_INDEX["left_shoulder"]], points_xyz[H36M_INDEX["left_elbow"]])
    angles["right_shoulder"] = angle_three_points(neck, points_xyz[H36M_INDEX["right_shoulder"]], points_xyz[H36M_INDEX["right_elbow"]])
    # Hips
    angles["left_hip"] = joint_angle_triplet(points_xyz, "spine", "left_hip", "left_knee")
    angles["right_hip"] = joint_angle_triplet(points_xyz, "spine", "right_hip", "right_knee")
    # Knees
    angles["left_knee"] = joint_angle_triplet(points_xyz, "left_hip", "left_knee", "left_ankle")
    angles["right_knee"] = joint_angle_triplet(points_xyz, "right_hip", "right_knee", "right_ankle")
    return angles


def compute_bodyline_deviation(points_xyz: np.ndarray) -> Dict[str, float]:
    """
    Compute body-line deviation: global Z vs line from mid-wrists to mid-ankles.
    """
    lw = points_xyz[H36M_INDEX["left_wrist"]]
    rw = points_xyz[H36M_INDEX["right_wrist"]]
    la = points_xyz[H36M_INDEX["left_ankle"]]
    ra = points_xyz[H36M_INDEX["right_ankle"]]
    mid_wrists = 0.5 * (lw + rw)
    mid_ankles = 0.5 * (la + ra)
    line = mid_wrists - mid_ankles
    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    deviation_deg = vector_angle(normalize_vector(line), z_axis)
    return {"bodyline_deg": float(deviation_deg)}


def compute_sway_metrics(hip_centers_aligned_xy: np.ndarray, fps: float) -> Dict[str, float]:
    """
    Sway metrics: RMS of XY displacement and a simple frequency proxy using zero-crossings on X.
    hip_centers_aligned_xy: (T, 2)
    """
    xy = hip_centers_aligned_xy
    rms = float(np.sqrt(np.mean(np.sum(xy**2, axis=1))))
    # frequency proxy: zero crossings along X
    x = xy[:, 0]
    zc = np.where(np.diff(np.signbit(x)))[0]
    freq_hz = float(len(zc) * 0.5 / (len(x) / max(fps, 1e-6))) if len(x) > 1 else 0.0
    return {"sway_rms_xy": rms, "sway_freq_hz": freq_hz}


def summarize_sequence_metrics(aligned_seq: np.ndarray, fps: float) -> Dict[str, float]:
    """
    Summarize metrics over a sequence.
    Returns per-joint angle mean/std and body-line + sway summaries.
    """
    T = aligned_seq.shape[0]
    angles_per_frame: Dict[str, List[float]] = {}
    # track hip center after alignment (should be near origin); use pelvis midpoint
    hip_xy = []
    for t in range(T):
        frame = aligned_seq[t]
        # pelvis mid as proxy for hip center in aligned space
        hip_center = 0.5 * (frame[H36M_INDEX["left_hip"]] + frame[H36M_INDEX["right_hip"]])
        hip_xy.append(hip_center[:2])
        ang = compute_core_angles(frame)
        for k, v in ang.items():
            angles_per_frame.setdefault(k, []).append(float(v))
    hip_xy = np.stack(hip_xy, axis=0)
    metrics: Dict[str, float] = {}
    for name, values in angles_per_frame.items():
        arr = np.array(values, dtype=float)
        metrics[f"{name}_mean"] = float(np.mean(arr))
        metrics[f"{name}_std"] = float(np.std(arr))
    metrics.update(compute_bodyline_deviation(np.mean(aligned_seq, axis=0)))
    metrics.update(compute_sway_metrics(hip_xy, fps))
    return metrics


