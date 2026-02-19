#!/bin/bash
# Generate a pileup pool chunk via HTCondor on lxplus.
# Usage: generate_pu_pool.sh <n_events> <seed> <output_file> [extra_args...]
#
# Environment variable PROJECT_DIR must be set (passed via condor .sub file).
set -eo pipefail

N_EVENTS="$1"
SEED="$2"
OUTPUT_FILE="$3"
shift 3
EXTRA_ARGS=("$@")

echo "=== generate_pu_pool.sh ==="
echo "N_EVENTS:    ${N_EVENTS}"
echo "SEED:        ${SEED}"
echo "OUTPUT_FILE: ${OUTPUT_FILE}"
echo "PROJECT_DIR: ${PROJECT_DIR}"
echo "EXTRA_ARGS:  ${EXTRA_ARGS[*]}"
echo "==========================="

# --- Environment setup ---
# LCG setup.sh has unbound variables, so disable nounset around it
set +u
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
set -u

# LCG ships Pythia 8.312 data but pip installs 8.317; unset to use pip version
unset PYTHIA8DATA

# Activate project venv (created with --system-site-packages on top of LCG)
source "${PROJECT_DIR}/.venv/bin/activate"

# --- Run ---
mkdir -p "$(dirname "${OUTPUT_FILE}")"

generate-pu-pool \
    -n "${N_EVENTS}" \
    --seed "${SEED}" \
    -o "${OUTPUT_FILE}" \
    "${EXTRA_ARGS[@]}"

echo "Done. Output: ${OUTPUT_FILE}"
