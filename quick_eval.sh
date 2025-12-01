#!/bin/bash
# Quick Evaluation Script for MemAgent Checkpoints
# Usage: bash quick_eval.sh [OPTIONS]

set -e

# Default values
OUTPUTS_DIR="/home/admin123/dl/MemAgent/outputs"
TEST_DATA="/home/admin123/dl/MemAgent/taskutils/memory_data/hotpotqa/hotpotqa_dev_20.parquet"
API_BASE="http://localhost:8000/v1"
API_KEY="EMPTY"
MAX_SAMPLES=100
CONCURRENCY=20
EXP_FILTER=""
CKPT_FILTER=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --exp_filter)
            EXP_FILTER="$2"
            shift 2
            ;;
        --ckpt_filter)
            CKPT_FILTER="$2"
            shift 2
            ;;
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --api_base)
            API_BASE="$2"
            shift 2
            ;;
        --concurrency)
            CONCURRENCY="$2"
            shift 2
            ;;
        --help)
            echo "Usage: bash quick_eval.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --exp_filter PATTERN     Filter experiments (e.g., '0.5B_kv', '7B')"
            echo "  --ckpt_filter PATTERN    Filter checkpoints (e.g., 'global_step_1000')"
            echo "  --max_samples N          Max test samples (default: 100)"
            echo "  --api_base URL           API endpoint (default: http://localhost:8000/v1)"
            echo "  --concurrency N          Concurrent requests (default: 20)"
            echo "  --help                   Show this help"
            echo ""
            echo "Examples:"
            echo "  # Evaluate all 0.5B_kv checkpoints"
            echo "  bash quick_eval.sh --exp_filter 0.5B_kv"
            echo ""
            echo "  # Evaluate specific checkpoint"
            echo "  bash quick_eval.sh --exp_filter 7B_kv --ckpt_filter global_step_1000"
            echo ""
            echo "  # Fast evaluation with limited samples"
            echo "  bash quick_eval.sh --max_samples 50 --concurrency 50"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage"
            exit 1
            ;;
    esac
done

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate memagent

# Build command
CMD="python3 batch_evaluate_deployed.py"
CMD="$CMD --outputs_dir $OUTPUTS_DIR"
CMD="$CMD --test_data $TEST_DATA"
CMD="$CMD --api_base $API_BASE"
CMD="$CMD --api_key $API_KEY"
CMD="$CMD --max_samples $MAX_SAMPLES"
CMD="$CMD --concurrency $CONCURRENCY"

if [ -n "$EXP_FILTER" ]; then
    CMD="$CMD --exp_filter $EXP_FILTER"
fi

if [ -n "$CKPT_FILTER" ]; then
    CMD="$CMD --checkpoint_filter $CKPT_FILTER"
fi

echo "==================================================================="
echo "MemAgent Checkpoint Evaluation"
echo "==================================================================="
echo "Outputs directory: $OUTPUTS_DIR"
echo "Test data:         $TEST_DATA"
echo "API base:          $API_BASE"
echo "Max samples:       $MAX_SAMPLES"
echo "Concurrency:       $CONCURRENCY"
[ -n "$EXP_FILTER" ] && echo "Experiment filter: $EXP_FILTER"
[ -n "$CKPT_FILTER" ] && echo "Checkpoint filter: $CKPT_FILTER"
echo "==================================================================="
echo ""
echo "Command: $CMD"
echo ""
echo "Starting evaluation..."
echo ""

# Run evaluation
$CMD

echo ""
echo "Evaluation complete! Check the generated JSON and markdown files."
