#!/bin/bash
# Pre-flight Check for 2-GPU 48GB LoRA Training

set -e

echo "========================================"
echo "Pre-flight Check: 2-GPU Setup"
echo "========================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check functions
check_pass() {
    echo -e "${GREEN}✓${NC} $1"
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

FAIL_COUNT=0

echo ""
echo "1. Checking GPU availability..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check GPU 6
GPU6_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 6 2>/dev/null || echo "0")
GPU6_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 6 2>/dev/null || echo "100")

if [ "$GPU6_MEM" -gt 40000 ]; then
    check_pass "GPU 6: ${GPU6_MEM}MB free (Util: ${GPU6_UTIL}%)"
elif [ "$GPU6_MEM" -gt 30000 ]; then
    check_warn "GPU 6: ${GPU6_MEM}MB free (Util: ${GPU6_UTIL}%) - Might be tight"
else
    check_fail "GPU 6: ${GPU6_MEM}MB free (Util: ${GPU6_UTIL}%) - NOT ENOUGH!"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# Check GPU 7
GPU7_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 7 2>/dev/null || echo "0")
GPU7_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 7 2>/dev/null || echo "100")

if [ "$GPU7_MEM" -gt 40000 ]; then
    check_pass "GPU 7: ${GPU7_MEM}MB free (Util: ${GPU7_UTIL}%)"
elif [ "$GPU7_MEM" -gt 30000 ]; then
    check_warn "GPU 7: ${GPU7_MEM}MB free (Util: ${GPU7_UTIL}%) - Might be tight"
else
    check_fail "GPU 7: ${GPU7_MEM}MB free (Util: ${GPU7_UTIL}%) - NOT ENOUGH!"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

echo ""
echo "2. Checking environment variables..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check PYTORCH_CUDA_ALLOC_CONF
if [ -z "${PYTORCH_CUDA_ALLOC_CONF}" ] || [ "${PYTORCH_CUDA_ALLOC_CONF}" = "" ]; then
    check_pass "PYTORCH_CUDA_ALLOC_CONF is not set (correct for vLLM v1)"
else
    if [[ "${PYTORCH_CUDA_ALLOC_CONF}" == *"expandable_segments"* ]]; then
        check_fail "PYTORCH_CUDA_ALLOC_CONF contains expandable_segments! Will cause vLLM v1 error"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    else
        check_pass "PYTORCH_CUDA_ALLOC_CONF is set but doesn't contain expandable_segments"
    fi
fi

echo ""
echo "3. Checking conda environment..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -z "${CONDA_DEFAULT_ENV}" ]; then
    check_warn "Conda environment not activated"
else
    if [ "${CONDA_DEFAULT_ENV}" = "memagent" ]; then
        check_pass "Conda environment: memagent (correct)"
    else
        check_warn "Conda environment: ${CONDA_DEFAULT_ENV} (expected: memagent)"
    fi
fi

echo ""
echo "4. Checking Ray status..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

RAY_PROCESSES=$(ps aux | grep -E "ray::" | grep -v grep | wc -l)
if [ "$RAY_PROCESSES" -gt 0 ]; then
    check_warn "Found ${RAY_PROCESSES} Ray processes running. Consider cleaning up with: pkill -f 'ray::'"
else
    check_pass "No old Ray processes found"
fi

echo ""
echo "5. Checking data files..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

TRAIN_FILE="/home/admin123/dl/MemAgent/taskutils/memory_data/hotpotqa/hotpotqa_train_1k.parquet"
VAL_FILE="/home/admin123/dl/MemAgent/taskutils/memory_data/hotpotqa/hotpotqa_dev_20.parquet"

if [ -f "$TRAIN_FILE" ]; then
    TRAIN_SIZE=$(du -h "$TRAIN_FILE" | cut -f1)
    check_pass "Training data: $TRAIN_FILE ($TRAIN_SIZE)"
else
    check_warn "Training data not found: $TRAIN_FILE (will be created)"
fi

if [ -f "$VAL_FILE" ]; then
    VAL_SIZE=$(du -h "$VAL_FILE" | cut -f1)
    check_pass "Validation data: $VAL_FILE ($VAL_SIZE)"
else
    check_fail "Validation data not found: $VAL_FILE"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

echo ""
echo "6. Checking model path..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

MODEL_PATH="/mnt/ssd2/models/Qwen2.5-7B-Instruct"
if [ -d "$MODEL_PATH" ]; then
    MODEL_SIZE=$(du -sh "$MODEL_PATH" 2>/dev/null | cut -f1 || echo "unknown")
    check_pass "Model path exists: $MODEL_PATH ($MODEL_SIZE)"
else
    check_fail "Model path not found: $MODEL_PATH"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

echo ""
echo "7. Checking disk space..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

OUTPUT_DIR="/home/admin123/dl/MemAgent/outputs"
DISK_FREE=$(df -h "$OUTPUT_DIR" | tail -1 | awk '{print $4}')
DISK_USAGE=$(df -h "$OUTPUT_DIR" | tail -1 | awk '{print $5}' | sed 's/%//')

if [ "$DISK_USAGE" -lt 90 ]; then
    check_pass "Output directory: $DISK_FREE free (${DISK_USAGE}% used)"
else
    check_warn "Output directory: $DISK_FREE free (${DISK_USAGE}% used) - Running low!"
fi

echo ""
echo "8. Configuration summary..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "GPUs: 6, 7"
echo "Tensor Parallel Size: 2"
echo "FSDP Size: 2"
echo "MAX_TOKEN_PER_GPU: 16000"
echo "vLLM GPU Memory: 40%"
echo "LoRA Rank: 64"
echo "Expected Peak Memory: ~32-37 GB per GPU"
echo "Experiment: lora_1k_2gpu_r64"

echo ""
echo "========================================"
if [ $FAIL_COUNT -eq 0 ]; then
    echo -e "${GREEN}✓ Pre-flight check PASSED${NC}"
    echo "Ready to start training!"
    echo ""
    echo "Run: bash run_memory_7B_lora.sh"
else
    echo -e "${RED}✗ Pre-flight check FAILED ($FAIL_COUNT issues)${NC}"
    echo "Please fix the issues above before training."
    exit 1
fi
echo "========================================"
