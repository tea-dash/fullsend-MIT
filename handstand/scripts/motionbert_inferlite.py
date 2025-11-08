import argparse
import os
import sys
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description="Lightweight MotionBERT inference without SMPL/visualization")
    parser.add_argument("--motionbert_dir", default=os.environ.get("MOTIONBERT_DIR", "models/motionbert_repo"))
    parser.add_argument("--config", default="configs/pose3d/MB_ft_h36m_global_lite.yaml")
    parser.add_argument("--ckpt", default="checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin")
    parser.add_argument("--json_path", required=True)
    parser.add_argument("--vid_path", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--clip_len", type=int, default=243)
    args = parser.parse_args()

    repo = os.path.abspath(args.motionbert_dir)
    # Prioritize MotionBERT repo over local scripts/lib to avoid name collision
    scripts_dir = os.path.dirname(__file__)
    if scripts_dir in sys.path:
        sys.path.remove(scripts_dir)
    sys.path.insert(0, repo)

    from lib.utils.tools import get_config  # type: ignore
    from lib.utils.learning import load_backbone  # type: ignore
    from lib.utils.utils_data import flip_data  # type: ignore
    from lib.data.dataset_wild import WildDetDataset  # type: ignore

    cfg = get_config(os.path.join(repo, args.config))
    model_backbone = load_backbone(cfg)
    if torch.cuda.is_available():
        model_backbone = nn.DataParallel(model_backbone).cuda()
    model_backbone.eval()

    ckpt_path = os.path.join(repo, args.ckpt)
    if not os.path.exists(ckpt_path):
        raise RuntimeError(
            f"MotionBERT checkpoint not found at {ckpt_path}.\n"
            "Please download the pose3d global lite weights and place them there."
        )
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model_backbone.load_state_dict(checkpoint["model_pos"], strict=True)
    model_pos = model_backbone

    os.makedirs(args.out_path, exist_ok=True)
    # Scale 2D to [-1,1] (per infer_wild default when pixel flag is False)
    dataset = WildDetDataset(args.json_path, clip_len=args.clip_len, scale_range=[1, 1], focus=None)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    preds: List[np.ndarray] = []
    with torch.no_grad():
        for batch_input in loader:
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()
            if cfg.no_conf:
                batch_input = batch_input[:, :, :, :2]
            if cfg.flip:
                batch_input_flip = flip_data(batch_input)
                p1 = model_pos(batch_input)
                p2 = model_pos(batch_input_flip)
                p2 = flip_data(p2)
                p = (p1 + p2) / 2.0
            else:
                p = model_pos(batch_input)
            if cfg.rootrel:
                p[:, :, 0, :] = 0
            else:
                p[:, 0, 0, 2] = 0
            if cfg.gt_2d:
                p[..., :2] = batch_input[..., :2]
            preds.append(p.cpu().numpy())
    results = np.hstack(preds)
    results = np.concatenate(results)
    np.save(os.path.join(args.out_path, "X3D.npy"), results)
    print(f"Saved 3D poses to {os.path.join(args.out_path, 'X3D.npy')}")


if __name__ == "__main__":
    main()


