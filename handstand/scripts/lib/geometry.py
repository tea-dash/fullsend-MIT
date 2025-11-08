import numpy as np


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        return np.zeros_like(vector)
    return vector / norm


def canonical_axes(left_hip: np.ndarray, right_hip: np.ndarray, neck: np.ndarray):
    """
    Build body-centric axes:
      - X: from left hip to right hip (pelvis left→right)
      - Z: from hip center to neck (toward head)
      - Y: cross(Z, X), then re-orthogonalize X = cross(Y, Z)
    Returns origin (hip center) and rotation matrix R (columns are axes).
    """
    hip_center = 0.5 * (left_hip + right_hip)
    x_axis = normalize_vector(right_hip - left_hip)
    z_axis = normalize_vector(neck - hip_center)
    y_axis = normalize_vector(np.cross(z_axis, x_axis))
    x_axis = normalize_vector(np.cross(y_axis, z_axis))
    R = np.stack([x_axis, y_axis, z_axis], axis=1)  # columns
    return hip_center, R


def transform_points(points_xyz: np.ndarray, origin: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Apply body-centric transform: p' = R^T @ (p - origin)
    points_xyz: (..., 3)
    """
    centered = points_xyz - origin
    return centered @ R  # since R columns are axes, R is orthonormal, R^T == R^-1


def angle_three_points(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Angle at point b formed by segments ba and bc in degrees.
    """
    v1 = a - b
    v2 = c - b
    v1 = normalize_vector(v1)
    v2 = normalize_vector(v2)
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def vector_angle(u: np.ndarray, v: np.ndarray) -> float:
    u = normalize_vector(u)
    v = normalize_vector(v)
    dot = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def mid_point(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return 0.5 * (p + q)


