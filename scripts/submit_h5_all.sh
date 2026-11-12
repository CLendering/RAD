#!/usr/bin/env bash
# Usage: bash scripts/submit_h5_all.sh [gpu_h100|gpu_a100] [datasets...]
set -euo pipefail

PARTITION="${1:-gpu_h100}"
shift $(( $# > 0 ? 1 : 0 ))
if [[ "${PARTITION}" != gpu_h100 && "${PARTITION}" != gpu_a100 ]]; then
  echo "Partition must be gpu_h100 or gpu_a100" >&2
  exit 2
fi

if (( $# )); then
  DATASETS=( "$@" )
else
  DATASETS=( mvtec visa realiad realiad_variety adnet manta_tiny )
fi
PROJECT="${RAD_PROJECT:-/projects/0/prjs2349}"
mkdir -p "${PROJECT}/rad_results"
cd "${PROJECT}/RAD"

echo "Submitting uniform 448x448 bicubic, no-crop RAD HDF5 experiments."
echo "Bank and eval jobs for each dataset use a SLURM afterok dependency."
for ds in "${DATASETS[@]}"; do
  case "${ds}" in
    mvtec|visa|realiad|manta_tiny|realiad_variety|adnet) ;;
    *) echo "Unknown dataset: ${ds}" >&2; exit 3;;
  esac
  bank=$(sbatch --parsable --partition="${PARTITION}" \
        --export=ALL,DATASET="${ds}",STAGE=bank,RAD_DIRECT_448=1 \
        scripts/rad_h5.sbatch)
  eval=$(sbatch --parsable --partition="${PARTITION}" \
        --dependency=afterok:"${bank}" \
        --export=ALL,DATASET="${ds}",STAGE=eval,RAD_DIRECT_448=1 \
        scripts/rad_h5.sbatch)
  echo "${ds}: bank=${bank} eval=${eval}"
done
