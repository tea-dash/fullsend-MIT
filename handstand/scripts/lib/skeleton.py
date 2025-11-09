from typing import List, Tuple

# COCO-17 edges (pairs of indices) for drawing skeletons
# Index mapping must match COCO17_NAMES order from joints.py
COCO17_EDGES: List[Tuple[int, int]] = [
    (5, 6),   # shoulders
    (5, 7), (7, 9),   # left arm
    (6, 8), (8, 10),  # right arm
    (11, 12),         # hips
    (5, 11), (6, 12), # torso
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]

# H36M-17 edges for MotionBERT outputs
H36M_EDGES: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3),       # right leg
    (0, 4), (4, 5), (5, 6),       # left leg
    (0, 7), (7, 8),               # pelvis->spine->neck
    (8, 10), (8, 9),              # neck->head_top, neck->nose
    (8, 11), (11, 12), (12, 13),  # left arm
    (8, 14), (14, 15), (15, 16),  # right arm
]


