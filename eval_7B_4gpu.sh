#!/bin/bash
# Quick Evaluation Script for 7B 4-GPU Training Results
# Usage:
#   bash eval_7B_4gpu.sh                    # Evaluate all checkpoints
#   bash eval_7B_4gpu.sh --step 2000        # Evaluate single checkpoint
#   bash eval_7B_4gpu.sh --merge-only       # Only merge checkpoints
#   bash eval_7B_4gpu.sh --eval-only        # Only evaluate (skip merge)
#   bash eval_7B_4gpu.sh --length 200       # Evaluate with 200 docs

set -eo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse custom eval length if provided
CUSTOM_LENGTH=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --length)
            CUSTOM_LENGTH="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Load configuration
CONFIG_FILE="${SCRIPT_DIR}/eval_config_7B_4gpu.rc"
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Error: Configuration file not found: ${CONFIG_FILE}"
    exit 1
fi

source "${CONFIG_FILE}"

# Override eval length if specified
if [ -n "${CUSTOM_LENGTH}" ]; then
    export EVAL_LENGTH="${CUSTOM_LENGTH}"
    echo "Using custom evaluation length: ${EVAL_LENGTH}"
fi

# Run evaluation script with config
exec bash "${SCRIPT_DIR}/run_evaluation.sh" --config eval_config_7B_4gpu.rc "$@"
