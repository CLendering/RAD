"""Compare RAD baseline and chunk-tuned evaluation using saved predictions."""
import argparse
import json
from pathlib import Path

import numpy as np

NAMES = ("I-AUROC", "I-AP", "I-F1", "P-AUROC", "P-AP", "P-F1", "P-AUPRO")


def inspect(directory):
    directory = Path(directory)
    metrics = json.loads((directory / "metrics.json").read_text())
    state = json.loads((directory / "spool" / "metric-state.json").read_text())
    if state["appended"] != state["num_samples"]:
        raise ValueError(f"Incomplete metrics at {directory}")
    n, h, w = state["num_samples"], state["height"], state["width"]
    maps = np.memmap(directory / "spool" / "maps.bin", dtype=np.dtype(state["map_dtype"]),
                     mode="r", shape=(n, h, w))
    scores = np.memmap(directory / "spool" / "scores.bin", dtype=np.dtype(state["score_dtype"]),
                       mode="r", shape=(n,))
    labels = np.memmap(directory / "spool" / "labels.bin", dtype=np.uint8, mode="r", shape=(n,))
    return metrics, maps, scores, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    bm, bmaps, bscores, blabels = inspect(args.baseline)
    cm, cmaps, cscores, clabels = inspect(args.candidate)
    if bmaps.shape != cmaps.shape:
        raise ValueError(f"Map shape mismatch: {bmaps.shape} vs {cmaps.shape}")
    assert np.array_equal(blabels, clabels), "Image labels differ"
    maxima, means = [], []
    for start in range(0, len(bmaps), 4):
        x = np.asarray(bmaps[start:start+4]).astype(np.float64)
        y = np.asarray(cmaps[start:start+4]).astype(np.float64)
        diff = np.abs(x-y)
        maxima.append(float(diff.max()))
        means.append(float(diff.mean()))
    max_map_diff = max(maxima)
    mean_map_diff = float(np.mean(means))
    max_score_diff = float(np.max(np.abs(np.asarray(bscores).astype(np.float64) -
                                         np.asarray(cscores).astype(np.float64))))
    metric_deltas = np.abs(np.asarray(bm["values"]) - np.asarray(cm["values"]))
    print(f"Samples: {len(bmaps)}")
    print(f"Maximum absolute anomaly-map difference: {max_map_diff:.9g}")
    print(f"Mean absolute anomaly-map difference: {mean_map_diff:.9g}")
    print(f"Maximum absolute image-score difference: {max_score_diff:.9g}")
    print("Metric comparison (reported percentage points):")
    for name, a, b in zip(NAMES, bm["values"], cm["values"]):
        print(f"  {name:<10} baseline={100*a:.4f} tuned={100*b:.4f} delta={100*(b-a):+.4f}")
    bprof, cprof = bm.get("retrieval_profile",{}), cm.get("retrieval_profile",{})
    for name in ("image_retrieval_seconds", "patch_matching_seconds"):
        before, after = bprof.get(name), cprof.get(name)
        if before and after:
            speedup = before/after
            print(f"{name}: baseline={before:.3f}s, tuned={after:.3f}s, speedup={speedup:.2f}x")
    if max_map_diff == 0 and max_score_diff == 0:
        print("Status: predictions are bit-identical.")
    elif max_map_diff <= 1e-4 and max_score_diff <= 1e-4 and metric_deltas.max() <= 1e-4:
        print("Status: predictions agree within small float32 tolerance (not bit-identical).")
    else:
        print("WARNING: Differences exceed conservative tolerance. Investigate before replacing baseline.")


if __name__ == "__main__":
    main()
