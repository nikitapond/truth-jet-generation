#!/bin/bash
# HTCondor wrapper for generate-pu-pool
# Usage: generate_pu_pool.sh <venv_path> <output_dir> <n_events> <seed> <ecm> <batch_size>
set -eo pipefail

VENV_PATH="$1"
OUTPUT_DIR="$2"
N_EVENTS="$3"
SEED="$4"
ECM="$5"
BATCH_SIZE="$6"

# Source LCG view for Python 3.11+ and system libraries
# (disable nounset — LCG setup.sh uses unbound variables)
set +u
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
source "${VENV_PATH}/bin/activate"
set -u

# Unset PYTHIA8DATA so pythia8mc uses its own bundled XML (avoids
# version mismatch with the older Pythia shipped in the LCG view)
export PYTHIA8DATA=""

OUTFILE="${OUTPUT_DIR}/chunk_$(printf '%04d' "${SEED}").h5"
mkdir -p "${OUTPUT_DIR}"

echo "Starting generate-pu-pool: seed=${SEED}, n=${N_EVENTS}, ecm=${ECM}, output=${OUTFILE}"
generate-pu-pool -n "${N_EVENTS}" -o "${OUTFILE}" --seed "${SEED}" --ecm "${ECM}" --batch-size "${BATCH_SIZE}"
echo "Done: ${OUTFILE}"
