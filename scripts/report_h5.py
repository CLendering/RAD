"""Summarise measured RAD bank size and image/patch retrieval costs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = ("I-AUROC", "I-AP", "I-F1", "P-AUROC", "P-AP", "P-F1", "P-AUPRO")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.bank / "manifest.json").read_text())
    files = sorted(p for p in args.bank.rglob("*") if p.is_file())
    size = sum(p.stat().st_size for p in files)
    result = {
        "bank": str(args.bank),
        "bank_bytes_measured": size,
        "bank_gb_measured": size / 1e9,
        "n_references": manifest["n_samples"],
        "layers": manifest["layers"],
        "feature_dtype": manifest["dtype"],
        "feature_files": {str(p.relative_to(args.bank)): p.stat().st_size for p in files},
    }
    if args.evaluation:
        summary = json.loads((args.evaluation / "summary.json").read_text())
        result["metrics"] = {
            **dict(zip(METRICS, summary["mean"])),
            "O-AUROC": summary.get("O-AUROC"),
            "mAD": summary["mAD"],
            "mAD_metrics": summary["mAD_metrics"],
        }
        # RAD's scores are in [0,1]. Provide ready-to-report percentages.
        result["metrics_percent"] = {
            key: None if value is None or key == "mAD_metrics" else value * 100
            for key, value in result["metrics"].items() if key != "mAD_metrics"
        }
        keys = ("profiled_queries", "image_retrieval_seconds",
                "patch_matching_seconds", "total_retrieval_seconds",
                "logical_cls_bytes_read", "logical_patch_bytes_read")
        profile = {key: 0 for key in keys}
        completed = 0
        for category in summary["categories"]:
            metrics_path = args.evaluation / "metrics" / category / "metrics.json"
            if not metrics_path.exists():
                raise FileNotFoundError(metrics_path)
            cat = json.loads(metrics_path.read_text())
            p = cat.get("retrieval_profile", {})
            for key in keys:
                profile[key] += p.get(key, 0)
            completed += int(cat.get("sample_count", 0))
        profile["evaluated_images"] = completed
        if profile["profiled_queries"]:
            profile["image_retrieval_ms_per_profiled_query"] = (
                profile["image_retrieval_seconds"] * 1000 / profile["profiled_queries"])
            profile["patch_matching_ms_per_profiled_query"] = (
                profile["patch_matching_seconds"] * 1000 / profile["profiled_queries"])
            profile["retrieval_ms_per_profiled_query"] = (
                profile["total_retrieval_seconds"] * 1000 / profile["profiled_queries"])
            profile["logical_feature_gb_per_profiled_query"] = (
                (profile["logical_cls_bytes_read"] + profile["logical_patch_bytes_read"])
                / profile["profiled_queries"] / 1e9)
        result["retrieval"] = profile
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
