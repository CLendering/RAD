#!/usr/bin/env bash
# After a dataset's default evaluation succeeds, run one K at a time on its
# existing full reference bank. No bank extraction or changed spatial radius.
# Usage: bash scripts/submit_k_sweep.sh adnet gpu_a100 32 256 900
set -euo pipefail

if (( $# < 3 )); then
  echo "Usage: $0 DATASET gpu_a100|gpu_h100 K [K ...]" >&2
  exit 2
fi
DATASET="$1"
PARTITION="$2"
shift 2
case "$PARTITION" in gpu_a100|gpu_h100) ;; *) echo "Unknown partition: $PARTITION" >&2; exit 3;; esac
case "$DATASET" in
  mvtec) ORIGINAL_K=150; R=1 ;;
  visa) ORIGINAL_K=900; R=2 ;;
  realiad) ORIGINAL_K=900; R=0 ;;
  manta_tiny) ORIGINAL_K=256; R=1 ;;
  realiad_variety) ORIGINAL_K=256; R=0 ;;
  adnet) ORIGINAL_K=256; R=1 ;;
  *) echo "Unknown dataset: $DATASET" >&2; exit 4 ;;
esac

PROJECT="${RAD_PROJECT:-/projects/0/prjs2349}"
SCRATCH="${RAD_SCRATCH:-/scratch-shared/${USER}/rad}"
PREPROC="bicubic448nocrop"
BANK="${SCRATCH}/banks/${DATASET}_dinov3_b16_${PREPROC}"
DEFAULT_OUT="${SCRATCH}/evaluations/${DATASET}_k${ORIGINAL_K}_r${R}_${PREPROC}"

if [[ ! -f "${BANK}/manifest.json" ]]; then
  echo "Bank is incomplete: ${BANK}" >&2
  exit 5
fi
if [[ ! -f "${DEFAULT_OUT}/summary.json" ]]; then
  echo "Default K=${ORIGINAL_K} evaluation is not complete: ${DEFAULT_OUT}" >&2
  echo "Wait until it succeeds. This safeguard avoids overlapping large metric spools." >&2
  exit 6
fi

cd "${PROJECT}/RAD"
last=""
for k in "$@"; do
  if ! [[ "${k}" =~ ^[1-9][0-9]*$ ]]; then
    echo "K must be a positive integer: ${k}" >&2
    exit 7
  fi
  if (( k == ORIGINAL_K )); then
    echo "Skipping existing default ${DATASET} K=${k}"
    continue
  fi
  dir="${SCRATCH}/evaluations/${DATASET}_k${k}_r${R}_${PREPROC}"
  if [[ -f "${dir}/summary.json" ]]; then
    echo "Already complete: ${dir}"
    continue
  fi
  dependency=()
  if [[ -n "${last}" ]]; then dependency=(--dependency="afterok:${last}"); fi
  job=$(sbatch --parsable \
    --partition="${PARTITION}" \
    "${dependency[@]}" \
    --export=ALL,DATASET="${DATASET}",STAGE=eval,RAD_DIRECT_448=1,RAD_K="${k}" \
    scripts/rad_h5.sbatch)
  echo "${DATASET} K=${k}: eval job ${job} (same bank, radius=${R})"
  last="${job}"
done
