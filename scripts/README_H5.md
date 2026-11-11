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
