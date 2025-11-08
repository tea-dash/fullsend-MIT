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


