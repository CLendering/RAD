"""CoBRA HDF5 input adapter for unmodified RAD inference/scoring.

Requires the CLendering/cobra repository on PYTHONPATH. RAD owns the
preprocessing (512 resize -> 448 center crop), features, retrieval, and metrics.
CoBRA is used ONLY for immutable HDF5 record lookup and image decoding.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset

from cobra.data import DATASETS, MULTI_VIEW, dataset_samples, object_ids
from cobra.packed import load_manifest, open_image, pack_folder

INPUT_SIZE = 448
_SAMPLE_CACHE = {}


def direct_448_preprocessing():
    """Provenance shared by the HDF5 bank builder and evaluator."""
    return {
        "image_size": 448, "crop_size": 448, "resize": "square",
        "interpolation": "bicubic",
        "order": ["Resize", "ToTensor", "Normalize"],
        "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
    }


def get_direct_448_transforms():
    """Match cobra.data.resize_image and its 448x448 pixel geometry.

    For pre-resized Real-IAD packs this resize is an identity operation.
    For MVTec, VisA, ADNet, and MANTA views, it matches CoBRA's bicubic resize.
    No center crop is applied. RAD's mask transform is retained.
    """
    from torchvision import transforms as T
    from torchvision.transforms import InterpolationMode
    image = T.Compose([
        T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    mask = T.Compose([T.Resize((448, 448)), T.ToTensor()])
    return image, mask


def manifest_for(dataset: str, root: str):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown CoBRA dataset {dataset!r}")
    return load_manifest(pack_folder(dataset, root, INPUT_SIZE) / "manifest.json")


def categories_for(dataset: str, root: str):
    rows = manifest_for(dataset, root)["splits"]["train"]["samples"]
    return sorted({row["category"] for row in rows})


def grouped_samples(dataset: str, root: str, split: str, categories: list[str]):
    key = (dataset, str(Path(root).resolve()), split)
    if key not in _SAMPLE_CACHE:
        rows = dataset_samples(dataset, root, split, INPUT_SIZE)
        by_category = defaultdict(list)
        for sample in rows:
            by_category[sample.category].append(sample)
        _SAMPLE_CACHE[key] = by_category
    by_category = _SAMPLE_CACHE[key]
    missing = set(categories) - set(by_category)
    if missing:
        raise ValueError(f"{dataset}/{split}: no samples for {sorted(missing)}")
    return [sample for category in categories for sample in by_category[category]]


def _identity(sample_ref):
    if sample_ref is None:
        return None
    return (sample_ref.file.sha256, sample_ref.offset, sample_ref.length)


class H5TrainDataset(Dataset):
    """Matches the (image,label) interface expected by RAD's IndexedDataset."""

    def __init__(self, samples, transform):
        self.samples = list(samples)
        self.transform = transform
        if any(int(s.label) != 0 or s.mask_path is not None for s in self.samples):
            raise ValueError("RAD reference bank must contain normal training observations only")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        with open_image(sample.image_path, sample.image_ref) as im:
            image = self.transform(im.convert("RGB"))
        return image, 0


class H5TestDataset(Dataset):
    """Matches RAD's (image, mask, image_label, original_path) interface."""

    def __init__(self, samples, transform, gt_transform, *, pack_sha256):
        self.samples = list(samples)
        self.transform = transform
        self.gt_transform = gt_transform
        self.img_paths = [s.image_path for s in self.samples]
        self.gt_paths = [s.mask_path for s in self.samples]
        self.labels = np.array([s.label for s in self.samples], dtype=np.int64)
        self.pack_sha256 = pack_sha256

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        with open_image(sample.image_path, sample.image_ref) as im:
            image = self.transform(im.convert("RGB"))
        if sample.mask_ref is not None:
            with open_image(sample.mask_path, sample.mask_ref) as mask:
                gt = self.gt_transform(mask.convert("L"))
        else:
            gt = torch.zeros((1, *image.shape[-2:]), dtype=torch.float32)
        if gt.shape[-2:] != image.shape[-2:]:
            raise ValueError(f"Image/mask dimensions differ: {sample.image_path}")
        return image, gt, int(sample.label), sample.image_path

    def h5_signature(self):
        """Stable provenance without hashing nonexistent legacy loose image paths."""
        digest = hashlib.sha256()
        for s in self.samples:
            row = (s.image_path, int(s.label), s.mask_path,
                   _identity(s.image_ref), _identity(s.mask_ref))
            digest.update(json.dumps(row, ensure_ascii=False).encode("utf-8"))
            digest.update(b"\n")
        # Conservatively budget potential source-resolution decoding. The two
        # Real-IAD variants are already prepared to 448, MANTA views to 256.
        source_side = 448 if self.samples and (
            "/realiad_" in self.samples[0].image_path or
            "/realiadvariety_" in self.samples[0].image_path) else 8192
        return dict(sha256=digest.hexdigest(), count=len(self),
                    max_source_pixels=source_side**2,
                    h5_sha256=self.pack_sha256)


def make_training(dataset: str, root: str, categories: list[str], transform):
    from torch.utils.data import ConcatDataset
    groups = grouped_samples(dataset, root, "train", categories)
    by_category = defaultdict(list)
    for row in groups:
        by_category[row.category].append(row)
    corpus = ConcatDataset([H5TrainDataset(by_category[c], transform) for c in categories])
    records = {r["path"]: r for r in manifest_for(dataset, root)["splits"]["train"]["records"]}
    return corpus, groups, records


def h5_reference(sample, index: int, categories: list[str],
                 counts: list[int], records: dict):
    import bisect
    group = bisect.bisect_right(counts, index)
    first = 0 if group == 0 else counts[group-1]
    category = categories[group]
    if sample.category != category:
        raise RuntimeError("HDF5 samples and bank index order disagree")
    meta = records[sample.image_path]
    with open_image(sample.image_path, sample.image_ref) as image:
        width, height = image.size
    return {
        "index": index, "category": category, "category_index": group,
        "sample_index": index - first,
        "path": sample.image_path, "relative_path": sample.image_path,
        "size": sample.image_ref.length, "mtime_ns": 0,
        "sha256": meta["sha256"], "image_width": width, "image_height": height,
        "h5_sha256": sample.image_ref.file.sha256,
        "h5_offset": sample.image_ref.offset, "h5_length": sample.image_ref.length,
    }


def make_test(dataset: str, root: str, category: str, transform, gt_transform):
    samples = grouped_samples(dataset, root, "test", [category])
    info = manifest_for(dataset, root)["splits"]["test"]
    return H5TestDataset(samples, transform, gt_transform, pack_sha256=info["sha256"])


def object_auroc(dataset: str, samples, metric_dir: str, fraction=0.01):
    """Use CoBRA's object IDs and joint-top-pixel pooling on RAD's saved maps."""
    if dataset not in MULTI_VIEW:
        return None
    metric_dir = Path(metric_dir)
    state_path = metric_dir / "spool" / "metric-state.json"
    if not state_path.exists():
        raise FileNotFoundError("Object AUROC requires the retained RAD metric spool: " + str(state_path))
    state = json.loads(state_path.read_text())
    if state["appended"] != len(samples):
        raise ValueError("Incomplete RAD predictions: cannot calculate object AUROC")
    width, height = state["width"], state["height"]
    dtype = np.dtype(state["map_dtype"])
    maps = np.memmap(metric_dir / "spool" / "maps.bin", dtype=dtype,
                     mode="r", shape=(len(samples), height, width))
    groups = defaultdict(list)
    for index, identity in enumerate(object_ids(samples)):
        groups[identity].append(index)
    labels, scores = [], []
    counts = defaultdict(int)
    for indices in groups.values():
        counts[len(indices)] += 1
        values = np.asarray(maps[indices]).reshape(-1)
        k = max(1, int(values.size * fraction))
        selected = np.partition(values, len(values) - k)[-k:]
        scores.append(float(np.mean(selected, dtype=np.float64)))
        labels.append(int(any(samples[i].label for i in indices)))
    if len(set(labels)) < 2:
        raise ValueError("Object AUROC needs normal and anomalous objects")
    return float(roc_auc_score(labels, scores)), dict(
        objects=len(groups), views_per_object=dict(sorted(counts.items())))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--categories", action="store_true")
    args = parser.parse_args()
    for cat in categories_for(args.dataset, args.data_path):
        print(cat)
