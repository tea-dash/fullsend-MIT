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


