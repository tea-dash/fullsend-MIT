import glob
import json
import os
from typing import Dict, List

import numpy as np

from lib.io_utils import save_json


def aggregate_metrics(metrics_dir: str) -> Dict[str, Dict[str, float]]:
    files = sorted(glob.glob(os.path.join(metrics_dir, "*.json")))
    if not files:
        raise RuntimeError(f"No metrics found in {metrics_dir}")
    # Collect keys
    all_keys = set()
    for f in files:
        with open(f, "r", encoding="utf-8") as rf:
            m = json.load(rf)
            all_keys |= set(m.keys())
    out: Dict[str, Dict[str, float]] = {}
    for key in sorted(all_keys):
        values: List[float] = []
        for f in files:
            with open(f, "r", encoding="utf-8") as rf:
                m = json.load(rf)
                if key in m:
                    try:
                        values.append(float(m[key]))
                    except Exception:
                        pass
        if not values:
            continue
        arr = np.array(values, dtype=float)
        out[key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
        }
    return out


def main():
    metrics_dir = os.path.join("data", "pro", "metrics")
    baseline_path = os.path.join("data", "pro", "metrics", "baseline.json")
    baseline = aggregate_metrics(metrics_dir)
    save_json(baseline, baseline_path)
    print(f"Baseline saved to {baseline_path}")


if __name__ == "__main__":
    main()


