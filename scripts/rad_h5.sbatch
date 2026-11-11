#!/bin/bash
#SBATCH --job-name=rad-h5
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/projects/0/prjs2349/rad_results/rad-%x-%j.log

set -euo pipefail

PROJECT="${RAD_PROJECT:-/projects/0/prjs2349}"
REPO="${PROJECT}/RAD"
SCRATCH="${RAD_SCRATCH:-/scratch-shared/${USER}/rad}"
DATASET="${DATASET:?Set DATASET=mvtec,visa,realiad,manta_tiny,realiad_variety,adnet}"
STAGE="${STAGE:?Set STAGE=bank or eval}"
mkdir -p "${SCRATCH}/banks" "${SCRATCH}/evaluations" "${SCRATCH}/logs" "${PROJECT}/rad_results"

module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
source "${REPO}/.venv/bin/activate"
export PYTHONPATH="${PROJECT}/cobra-source:${PYTHONPATH:-}"
export COBRA_DATA_SCRATCH=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export RAD_PROFILE=1
cd "${REPO}"

case "${DATASET}" in
  mvtec) ROOT="${PROJECT}/datasets_h5/mvtec_anomaly_detection"; K=150; R=1 ;;
  visa) ROOT="${PROJECT}/datasets_h5/VisA_processed"; K=900; R=2 ;;
  realiad) ROOT="${PROJECT}/datasets_h5/Real_IAD/Real_IAD"; K=900; R=0 ;;
  manta_tiny) ROOT="${PROJECT}/datasets_h5/manta_tiny_views"; K=150; R=1 ;;
  realiad_variety) ROOT="${PROJECT}/datasets_h5/Real-IAD_Variety"; K=900; R=0 ;;
  adnet) ROOT="${PROJECT}/datasets_h5/ADNet"; K=150; R=1 ;;
  *) echo "Unknown dataset ${DATASET}" >&2; exit 2 ;;
esac
# Match CoBRA's already-prepared 448x448 inputs for the two Real-IAD variants.
# RAD_DIRECT_448=0 restores RAD's published 512->448 geometry as a separate run.
DIRECT_448="${RAD_DIRECT_448:-auto}"
if [[ "${DIRECT_448}" == auto ]]; then
  if [[ "${DATASET}" == realiad || "${DATASET}" == realiad_variety ]]; then
    DIRECT_448=1
  else
    DIRECT_448=0
  fi
fi
if [[ "${DIRECT_448}" != 0 && "${DIRECT_448}" != 1 ]]; then
  echo "RAD_DIRECT_448 must be 0, 1 or auto" >&2; exit 8
fi
if [[ "${DIRECT_448}" == 1 && "${DATASET}" != realiad && "${DATASET}" != realiad_variety ]]; then
  echo "Direct 448 only supports Real-IAD / Real-IAD Variety packs" >&2; exit 9
fi
if [[ "${DIRECT_448}" == 1 ]]; then
  PREPROC="direct448"
  RESIZE=448
  EVAL_EXTRA=(--h5-direct-448)
else
  PREPROC="resize512crop448"
  RESIZE=512
  EVAL_EXTRA=()
fi
WEIGHT="${RAD_WEIGHT:-${PROJECT}/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth}"
if [[ ! -f "${WEIGHT}" && -f "${REPO}/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth" ]]; then
  WEIGHT="${REPO}/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
fi
test -f "${WEIGHT}" || { echo "Missing DINOv3 checkpoint: ${WEIGHT}" >&2; exit 3; }
test -f "${ROOT}/h5_dataset/${DATASET}/manifest.json" || {
  # CoBRA packs for Real-IAD are suffixed _448.
  if [[ "${DATASET}" != realiad && "${DATASET}" != realiad_variety ]]; then
    echo "Missing HDF5 manifest under ${ROOT}" >&2; exit 4
  fi
}
mapfile -t CATEGORIES < <(python -m rad_h5 --dataset "${DATASET}" --data-path "${ROOT}" --categories)
test "${#CATEGORIES[@]}" -gt 0 || exit 5
BANK="${SCRATCH}/banks/${DATASET}_dinov3_b16_${PREPROC}"
OUT="${SCRATCH}/evaluations/${DATASET}_k${K}_r${R}_${PREPROC}"
mkdir -p "${SCRATCH}/logs"
echo "Dataset=${DATASET} Stage=${STAGE} Categories=${#CATEGORIES[@]} K=${K} R=${R} geometry=${PREPROC}"
echo "Host=$(hostname) CUDA=${CUDA_VISIBLE_DEVICES:-none} Commit=$(git rev-parse HEAD)"
nvidia-smi
date -Is

if [[ "${STAGE}" == bank ]]; then
  if [[ -f "${BANK}/manifest.json" ]]; then
    echo "Already complete: ${BANK}"
  else
    RESUME=()
    if [[ -f "${BANK}/.progress.json" ]]; then RESUME=(--resume-bank); fi
    /usr/bin/time -v -o "${SCRATCH}/logs/${DATASET}-bank-${SLURM_JOB_ID:-manual}.time" \
      python -u build_bank_multilayer.py \
        --h5-dataset "${DATASET}" \
        --data_path "${ROOT}" \
        --item_list "${CATEGORIES[@]}" \
        --encoder_name dinov3_vitb16 \
        --encoder_weight "${WEIGHT}" \
        --image_size "${RESIZE}" --crop_size 448 \
        --layer_idx_list 3 6 9 11 \
        --batch_size "${RAD_BANK_BATCH:-32}" --num_workers 4 \
        --bank_path "${BANK}" "${RESUME[@]}"
  fi
  python scripts/report_h5.py --bank "${BANK}" \
      --output "${PROJECT}/rad_results/${DATASET}-${PREPROC}-bank.json"
elif [[ "${STAGE}" == eval ]]; then
  test -f "${BANK}/manifest.json" || { echo "Bank is incomplete: ${BANK}" >&2; exit 6; }
  RESUME=()
  if [[ -d "${OUT}/metrics" ]]; then RESUME=(--resume-eval); fi
  if [[ ! -f "${OUT}/summary.json" ]]; then
    /usr/bin/time -v -o "${SCRATCH}/logs/${DATASET}-eval-${SLURM_JOB_ID:-manual}.time" \
      python -u rad_mvtec_visa_3dadam.py \
        --h5-dataset "${DATASET}" \
        --data_path "${ROOT}" \
        --bank_path "${BANK}" \
        --encoder_name dinov3_vitb16 \
        --encoder_weight "${WEIGHT}" \
        --save_dir "${SCRATCH}/evaluations" \
        --save_name "${DATASET}_k${K}_r${R}_${PREPROC}" \
        --k_image "${K}" --resize_mask 256 --max_ratio 0.01 \
        --vis_max 0 --use_positional_bank --pos_radius "${R}" \
        --batch_size "${RAD_EVAL_BATCH:-8}" --num_workers 4 \
        --item_list "${CATEGORIES[@]}" "${EVAL_EXTRA[@]}" "${RESUME[@]}"
  else
    echo "Evaluation already complete: ${OUT}"
  fi
  python scripts/report_h5.py --bank "${BANK}" --evaluation "${OUT}" \
      --output "${PROJECT}/rad_results/${DATASET}-${PREPROC}-evaluation.json"
  cp "${OUT}/summary.json" "${PROJECT}/rad_results/${DATASET}-${PREPROC}-summary.json"
else
  echo "STAGE must be bank or eval" >&2; exit 7
fi
date -Is
