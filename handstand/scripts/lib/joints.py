from typing import List, Dict

# COCO-17 keypoints in order (standard MMPose HRNet/COCO ordering)
COCO17_NAMES: List[str] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

COCO17_INDEX: Dict[str, int] = {name: i for i, name in enumerate(COCO17_NAMES)}


def neck_index_from_shoulders() -> int:
    """
    There is no explicit neck in COCO-17; callers should compute neck as
    the midpoint of left/right shoulders.
    """
    return -1

# Human3.6M 17-joint order used by MotionBERT outputs (see halpe2h36m mapping)
H36M_17_NAMES: List[str] = [
    "pelvis",          # 0
    "right_hip",       # 1
    "right_knee",      # 2
    "right_ankle",     # 3
    "left_hip",        # 4
    "left_knee",       # 5
    "left_ankle",      # 6
    "spine",           # 7
    "neck",            # 8
    "nose",            # 9
    "head_top",        # 10
    "left_shoulder",   # 11
    "left_elbow",      # 12
    "left_wrist",      # 13
    "right_shoulder",  # 14
    "right_elbow",     # 15
    "right_wrist",     # 16
]

H36M_INDEX: Dict[str, int] = {name: i for i, name in enumerate(H36M_17_NAMES)}


