# RAD + CoBRA HDF5 on Snellius

This branch preserves RAD's DINOv3 ViT-B/16 512→448 preprocessing,
FP32 layers 3/6/9/11, score_batch, KNN, smoothing, and seven metrics.
It adds CoBRA HDF5 **storage/input only** and the CoBRA 8th multi-view
object-level AUROC. It never trains CoBRA's codec or runs CoBRA matching.

All metrics in JSON are fractions [0,1], except percentage fields generated
by scripts/report_h5.py. All category means are unweighted.

## Install

```bash
cd /projects/0/prjs2349
git clone https://github.com/CLendering/cobra.git cobra-source
git clone https://github.com/CLendering/RAD.git RAD  # if needed
cd RAD
git fetch origin
git switch feat/cobra-h5-six-benchmarks
module purge
module load 2024 Python/3.12.3-GCCcore-13.3.0
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt h5py==3.14.0 huggingface_hub
export PYTHONPATH=/projects/0/prjs2349/cobra-source:${PYTHONPATH:-}
python scripts/preflight_h5.py
```

Download the original DINOv3 ViT-B/16 LVD-1689M checkpoint (SHA-256
73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c)
into /projects/0/prjs2349/weights, or set RAD_WEIGHT.

## Submit

```bash
mkdir -p /projects/0/prjs2349/rad_results /scratch-shared/$USER/rad/{banks,evaluations,logs}
cd /projects/0/prjs2349/RAD
bank=$(sbatch --parsable --export=ALL,DATASET=mvtec,STAGE=bank scripts/rad_h5.sbatch)
sbatch --dependency=afterok:$bank --export=ALL,DATASET=mvtec,STAGE=eval scripts/rad_h5.sbatch
```

Use DATASET=visa,realiad,manta_tiny,realiad_variety,adnet. The default
partition is gpu_h100; override with sbatch --partition=gpu_a100.

Published RAD recipes: mvtec k=150 r=1; visa k=900 r=2;
realiad k=900 r=0. For the three datasets absent from RAD's paper,
transferred (NOT published) recipes: manta_tiny k=150 r=1;
realiad_variety k=900 r=0; adnet k=150 r=1.

Bank files and evaluation spools reside on shared scratch; reports and
summaries are copied to project space. `--resume-bank` and `--resume-eval`
are selected automatically for interrupted runs with matching inputs.
Never clean metric spools of multi-view datasets before object AUROC.
Set RAD_PROFILE=1 to time image retrieval and patch matching with CUDA
synchronizations. Logical bank read bytes are not physical disk IO.

## Verification

```bash
cat /projects/0/prjs2349/rad_results/mvtec-evaluation.json
find /scratch-shared/$USER/rad/banks/mvtec_dinov3_b16_448 -name 'layer_*_patches.f32' -printf '%s %p\n'
```

RAD's original Python code is still used to extract DINOv3 features,
construct the memory bank, rank CLS neighbors, match patches, generate maps,
and compute its seven metrics. No per-category retrieval restrictions or
coresets are introduced. Object-level scoring uses exactly CoBRA's object
identifiers and the top 1% of concatenated view-map pixels.

## All six datasets: matched-input bicubic 448², no crop

The default `scripts/rad_h5.sbatch` protocol is now
`RAD_DIRECT_448=1` for **all six datasets**. Its bank builder and
evaluator use `--h5-direct-448`, recorded as `bicubic448nocrop` in
the bank and output paths. The image transform is exactly
`torchvision.transforms.Resize((448,448), BICUBIC)` followed by
`ToTensor()` and ImageNet `Normalize()`, with **no CenterCrop**.
This matches CoBRA's `cobra.data.resize_image(...,448)` for all
HDF5 samples. The preflight asserts bit-exact normalized image
tensors for selected examples. RAD's mask transform, model, patch
retrieval, smoothing, scoring, and original seven metrics are unchanged.

For pre-resized 448² Real-IAD and Real-IAD Variety HDF5 images,
Resize((448,448)) does not geometrically change the source. For MVTec,
VisA, ADNet and MANTA views, bicubic 448 is applied once. MANTA-Tiny's
CoBRA headline used 280² rather than 448², so do not claim this is the
same resolution as the published CoBRA headline for that dataset.

`RAD_DIRECT_448=0` can still be used to create completely separate
`resize512crop448` banks/results, but is not the default.

Run a GPU smoke test after the CPU preflight. This branch has not been
tested against Snellius GPUs by the assistant. Bank size across six
datasets is approximately 2.820 TB of FP32 feature payload; scratch
space for metric spools and external sort is additional.


## Performance diagnosis: VisA K=900, radius=2

Measured on A100 with the original bounded streaming implementation:
200 VisA/candle images; CLS retrieval 4.619 seconds, patch matching
2687.938 seconds, or 13.440 s/image in patch matching. Reference
reads: 216,760,320,000 logical patch bytes across the category.
These are not OS-level disk-read measurements.

The default `reference-chunk-size=8`,
`query-patch-chunk-size=32`, and `memory-workspace-mib=256`
produce many small positional matching operations. On an isolated
worktree, test 32 references / 128 query patches / 1024 MiB workspace
with `scripts/bench_visa_chunks.sbatch`; this is **only** a
performance-parameter change, preserving K=900, radius=2, exact FP32
references, and the RAD scoring implementation.

The benchmark compares all 200 VisA/candle 256x256 maps, image scores,
and seven metrics with the original completed category; it refuses a
different bank, image geometry, K, or spatial radius. Chunk changes
may give small floating-point differences from operation ordering.
Do not replace the original full run without measured speedup and
numerical equivalence.

To prepare without modifying the existing working tree while other
banks are being built:

```bash
cd /projects/0/prjs2349/RAD
git fetch origin feat/cobra-h5-six-benchmarks
git worktree add --detach ../RAD-bench origin/feat/cobra-h5-six-benchmarks
sbatch --partition=gpu_a100 \
  --export=ALL,RAD_BENCH_REPO=/projects/0/prjs2349/RAD-bench \
  ../RAD-bench/scripts/bench_visa_chunks.sbatch
```

For *future* evaluations only, the standard `scripts/rad_h5.sbatch`
now allows `RAD_REF_CHUNK=32,RAD_QUERY_CHUNK=128,RAD_WORKSPACE_MIB=1024`;
the defaults remain RAD's 8/32/256. Do not change the contents of the
working tree of running jobs, since incomplete banks and evaluation
spools validate their source signatures on resume.
