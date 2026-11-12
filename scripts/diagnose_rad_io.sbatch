#!/bin/bash
#SBATCH --job-name=rad-io-audit
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/0/prjs2349/rad_results/rad-io-%j.log

set -euo pipefail

PROJECT=/projects/0/prjs2349
SCRATCH=/scratch-shared/${USER}/rad
DATASET=${RAD_DATASET:?Set RAD_DATASET}
K=${RAD_K:-256}
R=${RAD_RADIUS:-1}
N=${RAD_N:-8}

case "${DATASET}" in
  mvtec) DATA="${PROJECT}/datasets_h5/mvtec_anomaly_detection" ;;
  visa) DATA="${PROJECT}/datasets_h5/VisA_processed" ;;
  realiad) DATA="${PROJECT}/datasets_h5/Real_IAD/Real_IAD" ;;
  manta_tiny) DATA="${PROJECT}/datasets_h5/manta_tiny_views" ;;
  realiad_variety) DATA="${PROJECT}/datasets_h5/Real-IAD_Variety" ;;
  adnet) DATA="${PROJECT}/datasets_h5/ADNet" ;;
  *) echo "Unknown RAD_DATASET=${DATASET}" >&2; exit 2 ;;
esac

module purge
module load 2024 Python/3.12.3-GCCcore-13.3.0
source "${PROJECT}/RAD/.venv/bin/activate"
export PYTHONPATH="${PROJECT}/RAD:${PROJECT}/cobra-source:${PYTHONPATH:-}"
export COBRA_DATA_SCRATCH=0
export RAD_PROFILE=0
cd "${PROJECT}/RAD"

WEIGHTS=${RAD_WEIGHT:-${PROJECT}/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth}
BANK="${SCRATCH}/banks/${DATASET}_dinov3_b16_bicubic448nocrop"
mkdir -p "${PROJECT}/rad_results"
test -f "${BANK}/manifest.json" || { echo "Incomplete bank: ${BANK}" >&2; exit 3; }
test -f "${WEIGHTS}" || { echo "Missing checkpoint: ${WEIGHTS}" >&2; exit 4; }
OUT="${PROJECT}/rad_results/rad-io-${DATASET}-${SLURM_JOB_ID}.json"

python "${PROJECT}/rad_results/diagnose_rad_io.py" \
  --dataset "${DATASET}" --root "${DATA}" --bank "${BANK}" \
  --checkpoint "${WEIGHTS}" --k "${K}" --radius "${R}" \
  --num-images "${N}" --batch-size 4 --output "${OUT}"
