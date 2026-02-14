#!/bin/bash
# HTCondor wrapper for generate-pu-pool
# Usage: generate_pu_pool.sh <venv_path> <output_dir> <n_events> <seed> <ecm> <batch_size>
set -euo pipefail

VENV_PATH="$1"
OUTPUT_DIR="$2"
N_EVENTS="$3"
SEED="$4"
ECM="$5"
BATCH_SIZE="$6"

# Source LCG view for Python 3.11+ and system libraries
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
source "${VENV_PATH}/bin/activate"

OUTFILE="${OUTPUT_DIR}/chunk_$(printf '%04d' "${SEED}").h5"
mkdir -p "${OUTPUT_DIR}"

echo "Starting generate-pu-pool: seed=${SEED}, n=${N_EVENTS}, ecm=${ECM}, output=${OUTFILE}"
generate-pu-pool -n "${N_EVENTS}" -o "${OUTFILE}" --seed "${SEED}" --ecm "${ECM}" --batch-size "${BATCH_SIZE}"
echo "Done: ${OUTFILE}"
