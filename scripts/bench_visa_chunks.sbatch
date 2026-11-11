#!/bin/bash
#SBATCH --job-name=rad-visa-chunks
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/0/prjs2349/rad_results/visa-chunks-%j.log
set -euo pipefail

PROJECT="${RAD_PROJECT:-/projects/0/prjs2349}"
SCRATCH="${RAD_SCRATCH:-/scratch-shared/${USER}/rad}"
REPO="${PROJECT}/RAD"
BANK="${SCRATCH}/banks/visa_dinov3_b16_bicubic448nocrop"
ROOT="${PROJECT}/datasets_h5/VisA_processed"
WEIGHT="${RAD_WEIGHT:-${PROJECT}/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth}"
test -f "${WEIGHT}" || { echo "Missing checkpoint: ${WEIGHT}" >&2; exit 2; }
test -f "${BANK}/manifest.json" || { echo "Missing completed VisA bank: ${BANK}" >&2; exit 3; }

module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
source "${REPO}/.venv/bin/activate"
export PYTHONPATH="${PROJECT}/cobra-source:${PYTHONPATH:-}"
export COBRA_DATA_SCRATCH=0
export RAD_PROFILE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd "${REPO}"

NAME="visa_candle_chunks_${RAD_REF_CHUNK:-32}x${RAD_QUERY_CHUNK:-128}_${RAD_WORKSPACE_MIB:-1024}"
echo "Benchmark ${NAME} on existing VisA bank"
nvidia-smi
/usr/bin/time -v -o "${SCRATCH}/logs/${NAME}-${SLURM_JOB_ID}.time" \
python -u rad_mvtec_visa_3dadam.py \
    --h5-dataset visa --h5-direct-448 \
    --data_path "${ROOT}" \
    --bank_path "${BANK}" \
    --encoder_name dinov3_vitb16 --encoder_weight "${WEIGHT}" \
    --save_dir "${SCRATCH}/evaluations" --save_name "${NAME}" \
    --k_image 900 --resize_mask 256 --max_ratio 0.01 \
    --use_positional_bank --pos_radius 2 --vis_max 0 \
    --batch_size 8 --num_workers 4 \
    --reference-chunk-size "${RAD_REF_CHUNK:-32}" \
    --query-patch-chunk-size "${RAD_QUERY_CHUNK:-128}" \
    --memory-workspace-mib "${RAD_WORKSPACE_MIB:-1024}" \
    --item_list VisA/candle

python scripts/compare_rad_chunks.py \
    --baseline "${SCRATCH}/evaluations/visa_k900_r2_bicubic448nocrop/metrics/VisA/candle" \
    --candidate "${SCRATCH}/evaluations/${NAME}/metrics/VisA/candle" \
    | tee "${PROJECT}/rad_results/${NAME}-comparison.txt"
