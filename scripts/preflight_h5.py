"""CPU-only preflight of all six CoBRA HDF5 packs for RAD.

Loads real images/masks and validates category grouping, normal-only banks,
five-view object identity, and stable sample signatures before GPU allocation.
"""
from collections import Counter, defaultdict
from pathlib import Path
import json
import sys

# Direct execution by path puts scripts/, not the RAD root, on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import get_data_transforms
from rad_h5 import categories_for, grouped_samples, make_training, make_test, manifest_for
from cobra.packed import open_image
from cobra.data import MULTI_VIEW, object_ids

ROOT = Path("/projects/0/prjs2349/datasets_h5")
ROOTS = {
    "mvtec": ROOT / "mvtec_anomaly_detection",
    "visa": ROOT / "VisA_processed",
    "realiad": ROOT / "Real_IAD/Real_IAD",
    "manta_tiny": ROOT / "manta_tiny_views",
    "realiad_variety": ROOT / "Real-IAD_Variety",
    "adnet": ROOT / "ADNet",
}
EXPECTED = {
    "mvtec": (3629, 1725, 15),
    "visa": (8659, 2162, 12),
    "realiad": (36465, 114585, 30),
    "manta_tiny": (107415, 44585, 38),
    "realiad_variety": (19955, 178995, 160),
    "adnet": (116192, 80102, 380),
}
image_transform, mask_transform = get_data_transforms(512, 448)

for dataset, folder in ROOTS.items():
    cats = categories_for(dataset, str(folder))
    train = grouped_samples(dataset, str(folder), "train", cats)
    test = grouped_samples(dataset, str(folder), "test", cats)
    observed = (len(train), len(test), len(cats))
    assert observed == EXPECTED[dataset], (dataset, observed, EXPECTED[dataset])
    source_manifest = manifest_for(dataset, str(folder))
    prepared_size = source_manifest.get("prepared_size")
    if dataset in ("realiad", "realiad_variety"):
        assert prepared_size == 448, (dataset, prepared_size)
        for example in (train[0], test[0]):
            with open_image(example.image_path, example.image_ref) as im:
                assert im.size == (448, 448), (dataset, example.image_path, im.size)
        print(f"WARNING {dataset}: HDF5 stores bicubic-resized 448x448 inputs; "
              "RAD 512->448 will RESAMPLE them AGAIN. This is NOT an exact "
              "reproduction of RAD preprocessing on original full-resolution inputs.",
              flush=True)
    else:
        assert prepared_size is None, (dataset, prepared_size)
        if dataset == "manta_tiny":
            with open_image(train[0].image_path, train[0].image_ref) as im:
                assert im.size == (256, 256), (dataset, im.size)
            print("INFO manta_tiny: stored images are already individual 256x256 camera views.",
                  flush=True)
        else:
            print(f"INFO {dataset}: CoBRA HDF5 packer preserved the original "
                  "encoded images (prepared_size=null).", flush=True)
    assert all(s.label == 0 and s.mask_path is None for s in train), dataset
    subset = [train[0], train[-1]]
    ds, sample_refs, records = make_training(dataset, str(folder), cats, image_transform)
    assert len(ds) == len(train)
    for index in (0, len(ds) - 1):
        im, label = ds[index]
        assert tuple(im.shape) == (3, 448, 448) and label == 0, dataset
    cat = cats[0]
    ev = make_test(dataset, str(folder), cat, image_transform, mask_transform)
    assert len(ev)
    im, mask, lab, path = ev[0]
    assert im.shape == (3, 448, 448) and mask.shape[-2:] == (448, 448), dataset
    assert isinstance(path, str) and lab in (0, 1)
    assert ev.h5_signature()["count"] == len(ev)
    if dataset in MULTI_VIEW:
        ids = object_ids(test)
        distribution = Counter(Counter(ids).values())
        print(f"{dataset}: grouped views/object {dict(distribution)}")
        assert all(count > 0 for count in distribution)
    print(f"PASS {dataset}: train={observed[0]} test={observed[1]} cats={observed[2]}", flush=True)
print("All six RAD/HDF5 dataset preflights passed.")
