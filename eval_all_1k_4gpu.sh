#!/bin/bash
# Quick Evaluation Script for all_1k_4gpu Training Results
#
# Usage:
#   bash eval_all_1k_4gpu.sh                    # Evaluate all checkpoints (step 100 and 200)
#   bash eval_all_1k_4gpu.sh --step 200         # Evaluate single checkpoint (step 200)
#   bash eval_all_1k_4gpu.sh --merge-only       # Only merge checkpoints without evaluation
#   bash eval_all_1k_4gpu.sh --eval-only        # Only evaluate (skip merge, requires merged models)
#   bash eval_all_1k_4gpu.sh --length 100       # Evaluate with 100 docs instead of default (50)
#   bash eval_all_1k_4gpu.sh --force-merge      # Force re-merge even if merged model exists
#   bash eval_all_1k_4gpu.sh --force-eval       # Force re-evaluate even if results exist
#
# Example workflows:
#   # Quick test on 50 docs
#   bash eval_all_1k_4gpu.sh --step 200 --length 50
#
#   # Full evaluation on 100 docs for both checkpoints
#   bash eval_all_1k_4gpu.sh --length 100
#
#   # Merge models first, then evaluate later
#   bash eval_all_1k_4gpu.sh --merge-only
#   bash eval_all_1k_4gpu.sh --eval-only --length 100

set -eo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse custom eval length if provided and build new args array
CUSTOM_LENGTH=""
NEW_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --length)
            CUSTOM_LENGTH="$2"
            shift 2
            ;;
        *)
            NEW_ARGS+=("$1")
            shift
            ;;
    esac
done

# Load configuration
CONFIG_FILE="${SCRIPT_DIR}/eval_config_all_1k_4gpu.rc"
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

# Run evaluation script with config and remaining args
exec bash "${SCRIPT_DIR}/run_evaluation.sh" --config eval_config_all_1k_4gpu.rc "${NEW_ARGS[@]}"
