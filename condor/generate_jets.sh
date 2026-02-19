#!/bin/bash
# Generate hard-scatter jets via HTCondor on lxplus.
# Usage: generate_jets.sh <process_flag> <process_val> <n_events> <seed> <output_file> [extra_args...]
#
# process_flag: --process or --pythia-card
# process_val:  e.g. "ttbar" or "/path/to/card.cmnd"
#
# Environment variable PROJECT_DIR must be set (passed via condor .sub file).
set -eo pipefail

PROCESS_FLAG="$1"
PROCESS_VAL="$2"
N_EVENTS="$3"
SEED="$4"
OUTPUT_FILE="$5"
shift 5
EXTRA_ARGS=("$@")

echo "=== generate_jets.sh ==="
echo "PROCESS:     ${PROCESS_FLAG} ${PROCESS_VAL}"
echo "N_EVENTS:    ${N_EVENTS}"
echo "SEED:        ${SEED}"
echo "OUTPUT_FILE: ${OUTPUT_FILE}"
echo "PROJECT_DIR: ${PROJECT_DIR}"
echo "EXTRA_ARGS:  ${EXTRA_ARGS[*]}"
echo "========================"

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

truthjets \
    "${PROCESS_FLAG}" "${PROCESS_VAL}" \
    -n "${N_EVENTS}" \
    --seed "${SEED}" \
    -o "${OUTPUT_FILE}" \
    "${EXTRA_ARGS[@]}"

echo "Done. Output: ${OUTPUT_FILE}"
