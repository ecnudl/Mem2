#!/bin/bash
# Quick Evaluation Script for 7B LoRA 4-GPU Fast Training Results
# This is a wrapper around run_evaluation.sh with LoRA-specific configuration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/eval_config_lora_4gpu_fast.rc"

# Parse arguments
MODE="full"
EVAL_LENGTH=""
CHECKPOINT_STEP=""

show_help() {
    cat << EOF
Evaluate 7B LoRA 4-GPU Fast Training Results (lora_4gpu_fast_15k_n2)

Usage: bash $(basename $0) [OPTIONS]

Options:
    --help, -h              Show this help message
    --length LENGTH         Evaluation length (50,100,200,400,800,1600,3200,6400)
                            Default: 100
    --step STEP            Evaluate single checkpoint (e.g., 100, 200, 300)
                            Default: evaluate all available checkpoints
    --merge-only           Only merge checkpoints, skip evaluation
    --eval-only            Skip merging, only evaluate (requires merged models)
    --force-merge          Force re-merge even if merged model exists
    --force-eval           Force re-evaluate even if results exist

Examples:
    # Evaluate all checkpoints at length 100
    bash $(basename $0)

    # Evaluate only step 100 at length 200
    bash $(basename $0) --step 100 --length 200

    # Only merge all checkpoints without evaluation
    bash $(basename $0) --merge-only

    # Re-evaluate step 100 (skip merge, force new eval)
    bash $(basename $0) --step 100 --eval-only --force-eval

    # Evaluate at different document lengths
    bash $(basename $0) --length 400    # Evaluate with 400 documents

Checkpoints available:
    - Check ${SCRIPT_DIR}/outputs/lora_4gpu_fast_15k_n2/global_step_*/

Training configuration:
    - Model: Qwen2.5-7B-Instruct
    - LoRA Rank: 64, Alpha: 32
    - Chunk Size: 2500
    - Max New Tokens: 1024
    - Max Context Length: 15000
    - Training GPUs: 0,2,3,4
    - Evaluation GPU: 1 (to avoid conflict)
    - GRPO Sampling: n=2
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            exit 0
            ;;
        --length)
            EVAL_LENGTH="$2"
            shift 2
            ;;
        --step)
            CHECKPOINT_STEP="$2"
            shift 2
            ;;
        --merge-only)
            MODE="merge-only"
            shift
            ;;
        --eval-only)
            MODE="eval-only"
            shift
            ;;
        --force-merge)
            export FORCE_MERGE="yes"
            shift
            ;;
        --force-eval)
            export FORCE_EVAL="yes"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if configuration file exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Error: Configuration file not found: ${CONFIG_FILE}"
    echo "Please run this script from the MemAgent root directory"
    exit 1
fi

echo "========================================="
echo "7B LoRA 4-GPU Fast Evaluation"
echo "========================================="
echo "Configuration: ${CONFIG_FILE}"
echo "Mode: ${MODE}"

# Load configuration
source "${CONFIG_FILE}"

# Override EVAL_LENGTH if specified
if [ -n "${EVAL_LENGTH}" ]; then
    echo "Overriding evaluation length: ${EVAL_LENGTH}"
    export EVAL_LENGTH="${EVAL_LENGTH}"
fi

# Check GPU availability
echo ""
echo "Checking GPU ${CUDA_VISIBLE_DEVICES}..."
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_MEMORY=$(nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader,nounits -i ${CUDA_VISIBLE_DEVICES} 2>/dev/null || echo "N/A")
    echo "GPU Status: ${GPU_MEMORY}"

    FREE_MEM=$(echo "${GPU_MEMORY}" | awk -F',' '{print $2}' | tr -d ' ')
    if [ -n "${FREE_MEM}" ] && [ "${FREE_MEM}" -lt 30000 ]; then
        echo "Warning: GPU ${CUDA_VISIBLE_DEVICES} has less than 30GB free memory"
        echo "This may cause OOM during vLLM loading. Consider using a different GPU."
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# Verify checkpoints exist
echo ""
echo "Verifying checkpoints..."
if [ ! -d "${CHECKPOINT_BASE}" ]; then
    echo "Error: Checkpoint directory not found: ${CHECKPOINT_BASE}"
    exit 1
fi

if [ -n "${CHECKPOINT_STEP}" ]; then
    CHECKPOINT_DIR="${CHECKPOINT_BASE}/global_step_${CHECKPOINT_STEP}"
    if [ ! -d "${CHECKPOINT_DIR}" ]; then
        echo "Error: Checkpoint not found: ${CHECKPOINT_DIR}"
        echo "Available checkpoints:"
        ls -d ${CHECKPOINT_BASE}/global_step_* 2>/dev/null || echo "  None found"
        exit 1
    fi
    echo "✓ Found checkpoint: global_step_${CHECKPOINT_STEP}"
    # Override CHECKPOINT_STEPS with the single step
    export CHECKPOINT_STEPS="${CHECKPOINT_STEP}"
else
    echo "Available checkpoints:"
    ls -d ${CHECKPOINT_BASE}/global_step_* 2>/dev/null || echo "  None found"

    # Auto-detect all checkpoints if CHECKPOINT_STEPS is not set or is default
    if [ "${CHECKPOINT_STEPS}" == "100" ]; then
        AUTO_STEPS=$(ls -d ${CHECKPOINT_BASE}/global_step_* 2>/dev/null | grep -oP 'global_step_\K\d+' | sort -n | tr '\n' ' ')
        if [ -n "${AUTO_STEPS}" ]; then
            echo "Auto-detected checkpoints: ${AUTO_STEPS}"
            export CHECKPOINT_STEPS="${AUTO_STEPS}"
        fi
    fi
fi

# Build run_evaluation.sh arguments
# Pass only the basename, not absolute path (run_evaluation.sh will resolve it)
CONFIG_BASENAME=$(basename "${CONFIG_FILE}")
EVAL_ARGS="--config ${CONFIG_BASENAME}"

if [ "${MODE}" == "merge-only" ]; then
    EVAL_ARGS="${EVAL_ARGS} --merge-only"
elif [ "${MODE}" == "eval-only" ]; then
    EVAL_ARGS="${EVAL_ARGS} --eval-only"
fi

if [ -n "${CHECKPOINT_STEP}" ]; then
    EVAL_ARGS="${EVAL_ARGS} --step ${CHECKPOINT_STEP}"
fi

echo ""
echo "========================================="
echo "Starting Evaluation Pipeline"
echo "========================================="
echo "Command: bash run_evaluation.sh ${EVAL_ARGS}"
echo ""

# Run evaluation
bash "${SCRIPT_DIR}/run_evaluation.sh" ${EVAL_ARGS}

EVAL_EXIT_CODE=$?

if [ ${EVAL_EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✓ Evaluation Complete!"
    echo "========================================="
    echo ""
    echo "Results saved to: ${RESULT_BASE}"
    echo "Merged models: ${MERGED_MODEL_BASE}"
    echo "Logs: ${LOG_DIR}"
    echo ""

    # Show quick summary if results exist
    if [ "${MODE}" != "merge-only" ]; then
        echo "Quick Summary:"
        for step in ${CHECKPOINT_STEPS}; do
            RESULT_FILE="${RESULT_BASE}/${MODEL_IDENTIFIER}_step${step}/eval_${EVAL_LENGTH}_${EVAL_API}.jsonl"
            if [ -f "${RESULT_FILE}" ]; then
                echo "  - Step ${step}: ${RESULT_FILE}"
            fi
        done

        SUMMARY_FILE="${RESULT_BASE}/summary_${MODEL_IDENTIFIER}_eval_${EVAL_LENGTH}_${EVAL_API}.json"
        if [ -f "${SUMMARY_FILE}" ]; then
            echo ""
            echo "Summary JSON: ${SUMMARY_FILE}"
        fi
    fi
else
    echo ""
    echo "========================================="
    echo "✗ Evaluation Failed (exit code: ${EVAL_EXIT_CODE})"
    echo "========================================="
    echo "Check logs in: ${LOG_DIR}"
    exit ${EVAL_EXIT_CODE}
fi
