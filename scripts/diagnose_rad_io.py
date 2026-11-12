"""Independent RAD I/O diagnostic; never modifies banks or evaluation spools.

Measures CoBRA-HDF5 image decoding, DINOv3 encoding, RAD bank reads and
matching separately. Times are descriptive, not full-dataset latency.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from dinov3.hub.backbones import load_dinov3_model
from rad_bank import BankReader, checkpoint_fingerprint
from rad_h5 import make_test, categories_for, get_direct_448_transforms, direct_448_preprocessing
from rad_retrieval import ChunkedRetriever, RetrievalConfig


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser(description="RAD HDF5 and reference-bank I/O diagnostic")
    parser.add_argument("--dataset", required=True,
        choices=("mvtec", "visa", "realiad", "manta_tiny", "realiad_variety", "adnet"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--radius", type=int, required=True)
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--reference-chunk-size", type=int, default=8)
    parser.add_argument("--query-patch-chunk-size", type=int, default=32)
    parser.add_argument("--memory-workspace-mib", type=int, default=256)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.num_images < 1 or args.batch_size < 1 or args.k < 1 or args.radius < 0:
        parser.error("num-images, batch-size, and k must be positive; radius must be nonnegative")
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        parser.error("Run this diagnostic inside a SLURM GPU allocation")

    bank = BankReader(args.bank)
    bank.validate_provenance({
        "encoder": {
            "name": "dinov3_vitb16",
            "checkpoint_sha256": checkpoint_fingerprint(args.checkpoint),
        },
        "preprocessing": direct_448_preprocessing(),
        "feature_dtype": "float32",
        "seed": 1,
    })

    categories = categories_for(args.dataset, args.root)
    category = args.category if hasattr(args, "category") and args.category else categories[0]
    image_transform, mask_transform = get_direct_448_transforms()
    ds = make_test(args.dataset, args.root, category, image_transform, mask_transform)
    count = min(args.num_images, len(ds))
    indices = np.linspace(0, len(ds)-1, count, dtype=int).tolist()

    source_seconds = 0.0
    images = []
    for index in indices:
        begin = time.perf_counter()
        tensor, _mask, _label, _path = ds[index]
        source_seconds += time.perf_counter() - begin
        images.append(tensor)

    read_times = defaultdict(float)
    read_counts = defaultdict(int)
    read_bytes = defaultdict(int)

    for kind in ("cls", "patches"):
        original = getattr(bank, "read_" + kind)
        def timed(*args_, _original=original, _kind=kind, **kwargs):
            begin = time.perf_counter()
            result = _original(*args_, **kwargs)
            read_times[_kind] += time.perf_counter() - begin
            read_counts[_kind] += 1
            read_bytes[_kind] += result.numel() * result.element_size()
            return result
        setattr(bank, "read_" + kind, timed)

    print(f"dataset={args.dataset} category={category} test_images={count} "
          f"reference_images={bank.n_samples} K={args.k} radius={args.radius}", flush=True)
    model = load_dinov3_model(
        "dinov3_vitb16", pretrained_weight_path=args.checkpoint
    ).to(device).eval()
    config = RetrievalConfig(
        reference_chunk_size=args.reference_chunk_size,
        query_patch_chunk_size=args.query_patch_chunk_size,
        cls_chunk_size=4096,
        workspace_mib=args.memory_workspace_mib,
        max_score_mib=128,
    )
    retriever = ChunkedRetriever(bank, device, config)
    encoder_seconds = 0.0
    retrieval_seconds = 0.0

    for first in range(0, count, args.batch_size):
        x = torch.stack(images[first:first+args.batch_size]).to(device)
        synchronize(device)
        t0 = time.perf_counter()
        groups = model.get_intermediate_layers(
            x, n=bank.layers, return_class_token=True, reshape=False, norm=True
        )
        synchronize(device)
        encoder_seconds += time.perf_counter() - t0
        patch = [g[0] for g in groups]
        cls = [g[1] for g in groups]
        synchronize(device)
        t0 = time.perf_counter()
        score, _topk = retriever.score_batch(
            patch, cls, args.k, True, args.radius
        )
        synchronize(device)
        retrieval_seconds += time.perf_counter() - t0
        del x, groups, patch, cls, score, _topk

    remaining = retrieval_seconds - read_times["cls"] - read_times["patches"]
    report = {
        "dataset": args.dataset,
        "category": category,
        "tested_images": count,
        "bank_reference_count": bank.n_samples,
        "K": args.k,
        "positional_radius": args.radius,
        "bank": str(args.bank),
        "source_decode_transform_ms_per_image": 1000*source_seconds/count,
        "encoder_ms_per_image": 1000*encoder_seconds/count,
        "retrieval_ms_per_image": 1000*retrieval_seconds/count,
        "cls_bank_read_ms_per_image": 1000*read_times["cls"]/count,
        "patch_bank_read_ms_per_image": 1000*read_times["patches"]/count,
        "retrieval_excluding_blocking_bank_reads_ms_per_image": 1000*remaining/count,
        "cls_bank_read_calls": read_counts["cls"],
        "patch_bank_read_calls": read_counts["patches"],
        "logical_cls_gb_read": read_bytes["cls"]/1e9,
        "logical_patch_gb_read": read_bytes["patches"]/1e9,
        "peak_cuda_allocated_gb": torch.cuda.max_memory_allocated()/1e9,
        "caveats": [
            "Only a deterministic small subset of one category was evaluated.",
            "Bank read timers measure synchronous CPU read calls; the remaining retrieval time includes CPU, transfer, and CUDA overhead, not GPU-only compute.",
            "Logical bytes read are not physical disk bytes.",
            "DataLoader prefetch, metrics, and full end-to-end evaluation are excluded.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
