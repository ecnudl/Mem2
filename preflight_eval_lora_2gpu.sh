#!/bin/bash
# Pre-flight Check for 7B LoRA 2-GPU Evaluation
# Run this before starting evaluation to verify configuration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/eval_config_7B_lora_2gpu.rc"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================="
echo "Pre-flight Check: 7B LoRA 2-GPU Evaluation"
echo "========================================="
echo ""

FAIL_COUNT=0
WARN_COUNT=0

check_pass() {
    echo -e "${GREEN}✓${NC} $1"
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    FAIL_COUNT=$((FAIL_COUNT + 1))
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
    WARN_COUNT=$((WARN_COUNT + 1))
}

# Check 1: Configuration file exists
echo "1. Checking configuration file..."
if [ -f "${CONFIG_FILE}" ]; then
    check_pass "Configuration file exists: ${CONFIG_FILE}"
    source "${CONFIG_FILE}"
else
    check_fail "Configuration file not found: ${CONFIG_FILE}"
    exit 1
fi
echo ""

# Check 2: Checkpoint directory
echo "2. Checking checkpoint directory..."
if [ -d "${CHECKPOINT_BASE}" ]; then
    check_pass "Checkpoint directory exists: ${CHECKPOINT_BASE}"

    # Check for checkpoints
    FOUND_CHECKPOINTS=$(ls -d ${CHECKPOINT_BASE}/global_step_* 2>/dev/null | wc -l)
    if [ ${FOUND_CHECKPOINTS} -gt 0 ]; then
        check_pass "Found ${FOUND_CHECKPOINTS} checkpoint(s):"
        for ckpt in ${CHECKPOINT_BASE}/global_step_*; do
            step=$(basename ${ckpt} | sed 's/global_step_//')
            if [ -d "${ckpt}/actor" ]; then
                echo "    ✓ Step ${step} (actor directory exists)"
            else
                echo "    ✗ Step ${step} (missing actor directory)"
                FAIL_COUNT=$((FAIL_COUNT + 1))
            fi
        done
    else
        check_fail "No checkpoints found in ${CHECKPOINT_BASE}"
    fi
else
    check_fail "Checkpoint directory not found: ${CHECKPOINT_BASE}"
fi
echo ""

# Check 3: Base model
echo "3. Checking base model..."
if [ -d "${BASE_MODEL}" ]; then
    check_pass "Base model exists: ${BASE_MODEL}"

    # Check for essential files
    if [ -f "${BASE_MODEL}/config.json" ]; then
        check_pass "Found config.json"
    else
        check_warn "Missing config.json in base model"
    fi
else
    check_fail "Base model not found: ${BASE_MODEL}"
fi
echo ""

# Check 4: GPU availability
echo "4. Checking GPU ${CUDA_VISIBLE_DEVICES}..."
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_INFO=$(nvidia-smi --query-gpu=index,name,memory.free,memory.total,utilization.gpu --format=csv,noheader,nounits -i ${CUDA_VISIBLE_DEVICES} 2>/dev/null)

    if [ -n "${GPU_INFO}" ]; then
        GPU_NAME=$(echo "${GPU_INFO}" | awk -F',' '{print $2}' | xargs)
        GPU_FREE=$(echo "${GPU_INFO}" | awk -F',' '{print $3}' | xargs)
        GPU_TOTAL=$(echo "${GPU_INFO}" | awk -F',' '{print $4}' | xargs)
        GPU_UTIL=$(echo "${GPU_INFO}" | awk -F',' '{print $5}' | xargs)

        check_pass "GPU ${CUDA_VISIBLE_DEVICES}: ${GPU_NAME}"
        echo "    - Memory: ${GPU_FREE}/${GPU_TOTAL} MB free"
        echo "    - Utilization: ${GPU_UTIL}%"

        if [ "${GPU_FREE}" -lt 30000 ]; then
            check_warn "GPU has less than 30GB free (${GPU_FREE}MB). May cause OOM."
        else
            check_pass "GPU has sufficient memory (${GPU_FREE}MB free)"
        fi

        if [ "${GPU_UTIL}" -gt 50 ]; then
            check_warn "GPU utilization is high (${GPU_UTIL}%). Other processes may be running."
        fi
    else
        check_fail "Cannot query GPU ${CUDA_VISIBLE_DEVICES}"
    fi
else
    check_warn "nvidia-smi not found. Cannot check GPU status."
fi
echo ""

# Check 5: Port availability
echo "5. Checking port ${VLLM_PORT}..."
if command -v lsof >/dev/null 2>&1; then
    if lsof -i :${VLLM_PORT} >/dev/null 2>&1; then
        check_warn "Port ${VLLM_PORT} is already in use. vLLM may fail to start."
        echo "    Run: lsof -i :${VLLM_PORT} to see which process is using it"
    else
        check_pass "Port ${VLLM_PORT} is available"
    fi
else
    check_warn "lsof not found. Cannot check port availability."
fi
echo ""

# Check 6: Conda environment
echo "6. Checking conda environment..."
if [ -n "${CONDA_DEFAULT_ENV}" ]; then
    if [ "${CONDA_DEFAULT_ENV}" = "${CONDA_ENV}" ]; then
        check_pass "Conda environment: ${CONDA_DEFAULT_ENV} (correct)"
    else
        check_warn "Current conda env: ${CONDA_DEFAULT_ENV}, expected: ${CONDA_ENV}"
    fi
else
    check_warn "Conda environment not activated"
fi
echo ""

# Check 7: Python packages
echo "7. Checking required Python packages..."
if command -v python >/dev/null 2>&1; then
    # Check vLLM
    if python -c "import vllm" 2>/dev/null; then
        VLLM_VERSION=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
        check_pass "vLLM installed (version: ${VLLM_VERSION})"
    else
        check_fail "vLLM not installed"
    fi

    # Check transformers
    if python -c "import transformers" 2>/dev/null; then
        check_pass "transformers installed"
    else
        check_fail "transformers not installed"
    fi
else
    check_fail "Python not found in PATH"
fi
echo ""

# Check 8: Disk space
echo "8. Checking disk space..."
MERGED_DIR_PARENT=$(dirname "${MERGED_MODEL_BASE}")
RESULT_DIR_PARENT=$(dirname "${RESULT_BASE}")

for dir in "${MERGED_DIR_PARENT}" "${RESULT_DIR_PARENT}"; do
    if [ -d "${dir}" ]; then
        DISK_FREE=$(df -h "${dir}" | tail -1 | awk '{print $4}')
        DISK_USAGE=$(df -h "${dir}" | tail -1 | awk '{print $5}' | sed 's/%//')

        if [ "${DISK_USAGE}" -lt 90 ]; then
            check_pass "${dir}: ${DISK_FREE} free"
        else
            check_warn "${dir}: ${DISK_FREE} free (${DISK_USAGE}% used)"
        fi
    fi
done
echo ""

# Check 9: Output directories
echo "9. Checking output directories..."
mkdir -p "${MERGED_MODEL_BASE}" 2>/dev/null && check_pass "Merged model directory: ${MERGED_MODEL_BASE}" || check_fail "Cannot create: ${MERGED_MODEL_BASE}"
mkdir -p "${RESULT_BASE}" 2>/dev/null && check_pass "Result directory: ${RESULT_BASE}" || check_fail "Cannot create: ${RESULT_BASE}"
mkdir -p "${LOG_DIR}" 2>/dev/null && check_pass "Log directory: ${LOG_DIR}" || check_fail "Cannot create: ${LOG_DIR}"
echo ""

# Check 10: Training configuration match
echo "10. Verifying evaluation config matches training..."
echo "    Training config (from run_memory_7B_lora.sh):"
echo "      - Chunk Size: 1536"
echo "      - Max New Tokens: 64"
echo ""
echo "    Evaluation config:"
echo "      - RECURRENT_CHUNK_SIZE: ${RECURRENT_CHUNK_SIZE}"
echo "      - RECURRENT_MAX_NEW: ${RECURRENT_MAX_NEW}"

if [ "${RECURRENT_CHUNK_SIZE}" = "1536" ] && [ "${RECURRENT_MAX_NEW}" = "64" ]; then
    check_pass "Evaluation config matches training config"
else
    check_fail "Evaluation config does NOT match training config!"
    echo "    This will cause incorrect evaluation results!"
fi
echo ""

# Summary
echo "========================================="
echo "Pre-flight Check Summary"
echo "========================================="

if [ ${FAIL_COUNT} -eq 0 ] && [ ${WARN_COUNT} -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo ""
    echo "Ready to start evaluation. Run:"
    echo "  bash eval_lora_2gpu.sh"
    exit 0
elif [ ${FAIL_COUNT} -eq 0 ]; then
    echo -e "${YELLOW}⚠ ${WARN_COUNT} warning(s) found${NC}"
    echo ""
    echo "You can proceed, but review the warnings above."
    echo "To start evaluation, run:"
    echo "  bash eval_lora_2gpu.sh"
    exit 0
else
    echo -e "${RED}✗ ${FAIL_COUNT} error(s) found${NC}"
    if [ ${WARN_COUNT} -gt 0 ]; then
        echo -e "${YELLOW}⚠ ${WARN_COUNT} warning(s) found${NC}"
    fi
    echo ""
    echo "Please fix the errors above before running evaluation."
    exit 1
fi
