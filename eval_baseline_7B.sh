#!/bin/bash
# Evaluate Baseline (Untrained) 7B Model for Comparison with Trained Models
# This script evaluates the raw Qwen2.5-7B-Instruct model without any training
# Usage: bash eval_baseline_7B.sh

set -eo pipefail

# ==================== Configuration ====================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Paths
MEMAGENT_ROOT="${SCRIPT_DIR}"
BASE_MODEL="/mnt/ssd2/models/Qwen2.5-7B-Instruct"
RESULT_BASE="${MEMAGENT_ROOT}/taskutils/memory_eval/results"
export DATAROOT="${MEMAGENT_ROOT}/taskutils/memory_data/hotpotqa"  # Export for Python script
LOG_DIR="${MEMAGENT_ROOT}/eval_logs_baseline_7B"

# Conda environment
CONDA_ENV="memagent"
CONDA_SH="${HOME}/anaconda3/etc/profile.d/conda.sh"

# GPU Configuration (use different GPU from training)
export CUDA_VISIBLE_DEVICES="7"

# vLLM Service Configuration
VLLM_PORT="8005"  # Use unique port for baseline evaluation
VLLM_HOST="127.0.0.1"
VLLM_MAX_MODEL_LEN="4096"
VLLM_GPU_MEMORY_UTIL="0.85"
VLLM_DTYPE="bfloat16"
VLLM_STARTUP_WAIT="90"

# Evaluation Configuration
# IMPORTANT: Must match your training configuration!
export RECURRENT_CHUNK_SIZE="1536"        # Must match training (export for Python script)
export RECURRENT_MAX_NEW="64"             # Must match training (export for Python script)
export RECURRENT_MAX_CONTEXT_LEN="120000" # Export for Python script

# Evaluation dataset
EVAL_LENGTH="100"                  # Options: 50, 100, 200, 400, 800, 1600, 3200, 6400
EVAL_API="recurrent"               # Options: recurrent, recurrent-boxed, boxed
EVAL_N_PROC="32"                   # Concurrent requests

# Model naming
MODEL_IDENTIFIER="7B_baseline"     # Identifier for result files
EVAL_MODEL_NAME="Qwen2.5-7B-Instruct"
SERVED_MODEL_NAME="${EVAL_MODEL_NAME}"

# Result paths
RESULT_DIR="${RESULT_BASE}/${MODEL_IDENTIFIER}"
RESULT_FILE="${RESULT_DIR}/eval_${EVAL_LENGTH}_${EVAL_API}.jsonl"

# Derived variables
export URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"  # Export for Python script
export API_KEY="memagent-eval-key"                 # Export for Python script

# ==================== Setup Environment ====================
mkdir -p "${RESULT_DIR}"
mkdir -p "${LOG_DIR}"

# Activate conda environment
if [ -f "${CONDA_SH}" ]; then
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV}" || {
        echo "Error: Failed to activate conda environment: ${CONDA_ENV}"
        exit 1
    }
    echo "Activated conda environment: ${CONDA_ENV}"
else
    echo "Warning: Conda not found, assuming environment is already active"
fi

# ==================== Helper Functions ====================
function log_section() {
    echo ""
    echo "========================================="
    echo "$1"
    echo "========================================="
}

function log_step() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

function get_primary_cuda_device() {
    local visible="${CUDA_VISIBLE_DEVICES:-0}"
    visible="${visible%%,*}"
    visible="${visible// /}"
    if [[ "${visible}" =~ ^[0-9]+$ ]]; then
        echo "${visible}"
    else
        echo "0"
    fi
}

function resolve_gpu_memory_utilization() {
    local desired="${1:-${VLLM_GPU_MEMORY_UTIL}}"
    local gpu_index
    gpu_index="$(get_primary_cuda_device)"

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "${desired}"
        return
    fi

    local query_output
    if ! query_output=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null); then
        echo "${desired}"
        return
    fi

    local line_number=$((gpu_index + 1))
    local stats_line
    stats_line=$(echo "${query_output}" | sed -n "${line_number}p")
    if [ -z "${stats_line}" ]; then
        echo "${desired}"
        return
    fi

    local total_mib
    local used_mib
    total_mib=$(echo "${stats_line}" | cut -d',' -f1 | tr -d ' ')
    used_mib=$(echo "${stats_line}" | cut -d',' -f2 | tr -d ' ')

    if [ -z "${total_mib}" ] || [ -z "${used_mib}" ]; then
        echo "${desired}"
        return
    fi

    local adjusted
    adjusted=$(python - <<PY
total=${total_mib}
used=${used_mib}
desired=${desired:-0.85}
free = max(total - used, 0)
margin = 0.05
min_ratio = 0.15
if total == 0:
    safe = desired
else:
    free_ratio = max(free / total - margin, min_ratio)
    safe = min(desired, free_ratio)
print(f"{safe:.2f}")
PY
)

    if [ -z "${adjusted}" ]; then
        echo "${desired}"
        return
    fi

    local free_mib=$((total_mib - used_mib))
    if [ "${adjusted}" != "${desired}" ]; then
        log_step "GPU${gpu_index} free memory ${free_mib} MiB / ${total_mib} MiB insufficient. Using gpu_memory_utilization=${adjusted}" >&2
    else
        log_step "GPU${gpu_index} free memory ${free_mib} MiB / ${total_mib} MiB. Using gpu_memory_utilization=${adjusted}" >&2
    fi

    echo "${adjusted}"
}

function start_vllm_service() {
    local gpu_util
    gpu_util=$(resolve_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTIL}")

    # Stop existing vLLM service
    if [ -f "${LOG_DIR}/vllm.pid" ]; then
        local old_pid=$(cat "${LOG_DIR}/vllm.pid")
        if kill -0 "${old_pid}" 2>/dev/null; then
            log_step "Stopping existing vLLM service (PID: ${old_pid})"
            kill "${old_pid}" 2>/dev/null || true
            sleep 5
        fi
    fi

    log_step "Starting vLLM service for baseline model: ${BASE_MODEL}"
    nohup vllm serve "${BASE_MODEL}" \
        --port "${VLLM_PORT}" \
        --host "${VLLM_HOST}" \
        --trust-remote-code \
        --dtype "${VLLM_DTYPE}" \
        --max-model-len "${VLLM_MAX_MODEL_LEN}" \
        --gpu-memory-utilization "${gpu_util}" \
        --disable-log-requests \
        --served-model-name "${SERVED_MODEL_NAME}" \
        > "${LOG_DIR}/vllm_baseline.log" 2>&1 &

    local vllm_pid=$!
    echo "${vllm_pid}" > "${LOG_DIR}/vllm.pid"

    log_step "vLLM PID: ${vllm_pid}, waiting ${VLLM_STARTUP_WAIT}s for startup..."
    sleep "${VLLM_STARTUP_WAIT}"

    # Check if service is running
    if ! kill -0 "${vllm_pid}" 2>/dev/null; then
        echo "Error: vLLM service failed to start. Check log: ${LOG_DIR}/vllm_baseline.log"
        return 1
    fi

    # Test service availability
    if ! curl -s "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" > /dev/null; then
        echo "Warning: vLLM service may not be ready yet. Waiting additional 30s..."
        sleep 30
    fi

    log_step "vLLM service is ready"
    return 0
}

function stop_vllm_service() {
    if [ -f "${LOG_DIR}/vllm.pid" ]; then
        local vllm_pid=$(cat "${LOG_DIR}/vllm.pid")
        if kill -0 "${vllm_pid}" 2>/dev/null; then
            log_step "Stopping vLLM service (PID: ${vllm_pid})"
            kill "${vllm_pid}" 2>/dev/null || true
            sleep 3
        fi
        rm -f "${LOG_DIR}/vllm.pid"
    fi
}

function run_evaluation() {
    if [ -f "${RESULT_FILE}" ]; then
        log_step "Evaluation results already exist: ${RESULT_FILE}"
        log_step "Delete the file to re-run evaluation"
        return 0
    fi

    log_step "Running evaluation for baseline 7B model"

    cd "${MEMAGENT_ROOT}/taskutils/memory_eval"

    python ruler_hqa.py \
        --length "${EVAL_LENGTH}" \
        --model "${EVAL_MODEL_NAME}" \
        --tokenizer "${BASE_MODEL}" \
        --save_dir "${RESULT_DIR}" \
        --save_file "eval_${EVAL_LENGTH}_${EVAL_API}" \
        --api "${EVAL_API}" \
        --n_proc "${EVAL_N_PROC}" \
        --url "${URL}" \
        --api_key "${API_KEY}" \
        2>&1 | tee "${LOG_DIR}/eval_baseline.log"

    cd "${MEMAGENT_ROOT}"

    log_step "Evaluation completed: ${RESULT_FILE}"
    return 0
}

function print_results() {
    log_section "Baseline Evaluation Results"

    if [ ! -f "${RESULT_FILE}" ]; then
        echo "No results found: ${RESULT_FILE}"
        return 1
    fi

    echo "Model: Baseline Qwen2.5-7B-Instruct (Untrained)"
    echo "Evaluation Length: ${EVAL_LENGTH} documents"
    echo ""
    echo "Metric  | Score"
    echo "--------|-------"

    local stats=$(python3 -c "
import json
scores = {'f1': [], 'em': [], 'sub_em': []}
try:
    with open('${RESULT_FILE}') as f:
        for line in f:
            item = json.loads(line)
            scores['f1'].append(item.get('judge_f1', 0))
            scores['em'].append(item.get('judge_em', 0))
            scores['sub_em'].append(item.get('judge_sub_em', 0))
    f1 = sum(scores['f1'])*100/len(scores['f1']) if scores['f1'] else 0
    em = sum(scores['em'])*100/len(scores['em']) if scores['em'] else 0
    sub_em = sum(scores['sub_em'])*100/len(scores['sub_em']) if scores['sub_em'] else 0
    print(f'F1      | {f1:.2f}%')
    print(f'EM      | {em:.2f}%')
    print(f'Sub-EM  | {sub_em:.2f}%')
except Exception as e:
    print('Error parsing results')
" 2>/dev/null)

    echo "${stats}"
    echo ""
    echo "Detailed results: ${RESULT_FILE}"
    echo "Logs: ${LOG_DIR}"
}

# ==================== Main Execution ====================
log_section "Baseline 7B Model Evaluation"
log_step "Model: ${BASE_MODEL}"
log_step "Evaluation Length: ${EVAL_LENGTH} documents"
log_step "GPU: ${CUDA_VISIBLE_DEVICES}"
log_step "vLLM Port: ${VLLM_PORT}"
log_step "Recurrent Config: chunk_size=${RECURRENT_CHUNK_SIZE}, max_new=${RECURRENT_MAX_NEW}"

# Trap to ensure cleanup on exit
trap 'stop_vllm_service' EXIT

# Start vLLM service
if ! start_vllm_service; then
    echo "Error: Failed to start vLLM service"
    exit 1
fi

# Run evaluation
if ! run_evaluation; then
    echo "Error: Evaluation failed"
    exit 1
fi

# Stop vLLM service
stop_vllm_service

# Print results
print_results

log_section "Evaluation Complete!"
echo ""
echo "Next Steps:"
echo "  1. Compare with trained model results in: ${RESULT_BASE}"
echo "  2. To evaluate at different lengths, edit EVAL_LENGTH in this script"
echo "  3. To evaluate LoRA model, use: bash run_evaluation.sh --config eval_config_7B_lora.rc"
