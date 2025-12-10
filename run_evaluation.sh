#!/bin/bash
# MemAgent One-Click Evaluation Script
# Usage: bash run_evaluation.sh [OPTIONS]
#   Options:
#     --config FILE       Use custom config file (default: eval_config.rc)
#     --step STEP         Evaluate single checkpoint step (e.g., 10000)
#     --steps "1000 2000" Evaluate multiple checkpoint steps
#     --merge-only        Only merge checkpoints, skip evaluation
#     --eval-only         Skip merging, only evaluate (requires merged models)
#     --help              Show this help message

set -eo pipefail  # Exit on error and fail on pipeline errors

# ==================== Parse Arguments ====================
CONFIG_FILE="eval_config.rc"
MODE="full"  # full, merge-only, eval-only
CUSTOM_STEPS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --step)
            CUSTOM_STEPS="$2"
            shift 2
            ;;
        --steps)
            CUSTOM_STEPS="$2"
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
        --help|-h)
            head -n 8 "$0" | tail -n 7
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ==================== Load Configuration ====================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/${CONFIG_FILE}"

if [ ! -f "${CONFIG_PATH}" ]; then
    echo "Error: Configuration file not found: ${CONFIG_PATH}"
    exit 1
fi

echo "Loading configuration from: ${CONFIG_PATH}"
source "${CONFIG_PATH}"

# Fallback defaults if config did not set these
: "${EVAL_MODEL_NAME:=Qwen2.5-1.5B-Instruct}"
: "${SERVED_MODEL_NAME:=${EVAL_MODEL_NAME}}"

# Override checkpoint steps if provided
if [ -n "${CUSTOM_STEPS}" ]; then
    CHECKPOINT_STEPS="${CUSTOM_STEPS}"
fi

# Handle SINGLE_CHECKPOINT variable
if [ -n "${SINGLE_CHECKPOINT}" ] && [ -z "${CUSTOM_STEPS}" ]; then
    CHECKPOINT_STEPS="${SINGLE_CHECKPOINT}"
fi

# Auto-detect MODEL_IDENTIFIER if not set
if [ -z "${MODEL_IDENTIFIER}" ]; then
    # Infer from checkpoint base directory name
    checkpoint_basename=$(basename "${CHECKPOINT_BASE}")
    # Extract model identifier from directory name (e.g., "1.5B_kv" from "/path/to/1.5B_kv")
    MODEL_IDENTIFIER="${checkpoint_basename}"

    # If inference fails or is generic, use a more specific pattern
    if [ -z "${MODEL_IDENTIFIER}" ] || [ "${MODEL_IDENTIFIER}" = "." ] || [ "${MODEL_IDENTIFIER}" = "/" ]; then
        MODEL_IDENTIFIER="model"
    fi

    echo "Auto-detected MODEL_IDENTIFIER: ${MODEL_IDENTIFIER}"
fi

# ==================== Setup Environment ====================
mkdir -p "${MERGED_MODEL_BASE}"
mkdir -p "${RESULT_BASE}"
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
    echo "Warning: Conda not found at ${CONDA_SH}, assuming environment is already active"
fi

# Change to project root
cd "${MEMAGENT_ROOT}"

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

TOKENIZER_FILES=("tokenizer.json" "tokenizer_config.json" "vocab.json" "merges.txt" "tokenizer.model" "special_tokens_map.json")

function ensure_tokenizer_files() {
    local target_dir=$1
    local source_dir="${BASE_MODEL}"
    local copied=0

    for fname in "${TOKENIZER_FILES[@]}"; do
        if [ -f "${source_dir}/${fname}" ] && [ ! -f "${target_dir}/${fname}" ]; then
            cp "${source_dir}/${fname}" "${target_dir}/"
            copied=1
        fi
    done

    if [ "${copied}" -eq 1 ]; then
        log_step "Synced tokenizer files to ${target_dir}"
    fi
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
desired=${desired:-0.60}
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
        log_step "GPU${gpu_index} free memory ${free_mib} MiB / ${total_mib} MiB insufficient for requested gpu_memory_utilization=${desired}. Using ${adjusted} instead." >&2
    else
        log_step "GPU${gpu_index} free memory ${free_mib} MiB / ${total_mib} MiB. Using gpu_memory_utilization=${adjusted}." >&2
    fi

    echo "${adjusted}"
}

function detect_lora_checkpoint() {
    local ckpt_dir=$1
    # Check if any checkpoint file contains LoRA structure
    for ckpt_file in "${ckpt_dir}"/model_world_size_*_rank_0.pt; do
        if [ -f "${ckpt_file}" ]; then
            # Quick check: does the checkpoint contain lora_A or base_model keys?
            python3 -c "
import torch
try:
    state_dict = torch.load('${ckpt_file}', map_location='cpu', weights_only=False)
    keys = list(state_dict.keys())
    has_lora = any('lora_A' in k or 'lora_B' in k or 'base_model' in k for k in keys[:100])
    exit(0 if has_lora else 1)
except Exception:
    exit(1)
" 2>/dev/null
            return $?
        fi
    done
    return 1
}

function merge_checkpoint() {
    local step=$1
    local ckpt_dir="${CHECKPOINT_BASE}/global_step_${step}/actor"
    local merged_dir="${MERGED_MODEL_BASE}/${MODEL_IDENTIFIER}_step${step}"

    if [ ! -d "${ckpt_dir}" ]; then
        echo "Warning: Checkpoint not found: ${ckpt_dir}, skipping..."
        return 1
    fi

    if [ -f "${merged_dir}/config.json" ] && [ "${FORCE_MERGE}" != "yes" ]; then
        log_step "Merged model already exists: ${merged_dir}"
        ensure_tokenizer_files "${merged_dir}"
        return 0
    fi

    log_step "Merging checkpoint: global_step_${step}"

    # Detect if this is a LoRA checkpoint
    local is_lora_ckpt=0
    if detect_lora_checkpoint "${ckpt_dir}"; then
        is_lora_ckpt=1
        log_step "Detected LoRA checkpoint - using LoRA merger"
    else
        log_step "Standard checkpoint - using FSDP merger"
    fi

    if [ ${is_lora_ckpt} -eq 1 ]; then
        # Use LoRA merger with LoRA parameters from config
        local lora_rank="${LORA_RANK:-64}"
        local lora_alpha="${LORA_ALPHA:-32}"
        log_step "LoRA parameters: rank=${lora_rank}, alpha=${lora_alpha}"

        python scripts/lora_merger.py \
            --hf_model_path "${BASE_MODEL}" \
            --local_dir "${ckpt_dir}" \
            --target_dir "${merged_dir}" \
            --lora_rank ${lora_rank} \
            --lora_alpha ${lora_alpha} \
            2>&1 | tee "${LOG_DIR}/merge_step${step}.log"
    else
        # Use standard FSDP merger
        python scripts/model_merger.py \
            --backend fsdp \
            --hf_model_path "${BASE_MODEL}" \
            --local_dir "${ckpt_dir}" \
            --target_dir "${merged_dir}" \
            2>&1 | tee "${LOG_DIR}/merge_step${step}.log"
    fi

    if [ $? -ne 0 ]; then
        echo "Error: Merge failed for step ${step}"
        return 1
    fi

    log_step "Merge completed: ${merged_dir}"
    ensure_tokenizer_files "${merged_dir}"
    return 0
}

function start_vllm_service() {
    local merged_dir=$1
    local step=$2
    local gpu_util
    gpu_util=$(resolve_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTIL}")
    local served_model_name="${SERVED_MODEL_NAME:-$(basename "${merged_dir}")}"

    # Stop existing vLLM service
    if [ -f "${LOG_DIR}/vllm.pid" ]; then
        local old_pid=$(cat "${LOG_DIR}/vllm.pid")
        if kill -0 "${old_pid}" 2>/dev/null; then
            log_step "Stopping existing vLLM service (PID: ${old_pid})"
            kill "${old_pid}" 2>/dev/null || true
            sleep 5
        fi
    fi

    log_step "Starting vLLM service for: ${merged_dir}"
    nohup vllm serve "${merged_dir}" \
        --port "${VLLM_PORT}" \
        --host "${VLLM_HOST}" \
        --trust-remote-code \
        --dtype "${VLLM_DTYPE}" \
        --max-model-len "${VLLM_MAX_MODEL_LEN}" \
        --gpu-memory-utilization "${gpu_util}" \
        --disable-log-requests \
        --served-model-name "${served_model_name}" \
        > "${LOG_DIR}/vllm_step${step}.log" 2>&1 &

    local vllm_pid=$!
    echo "${vllm_pid}" > "${LOG_DIR}/vllm.pid"

    log_step "vLLM PID: ${vllm_pid}, waiting ${VLLM_STARTUP_WAIT}s for startup (served id: ${served_model_name})..."
    sleep "${VLLM_STARTUP_WAIT}"

    # Check if service is running
    if ! kill -0 "${vllm_pid}" 2>/dev/null; then
        echo "Error: vLLM service failed to start. Check log: ${LOG_DIR}/vllm_step${step}.log"
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

function run_evaluation() {
    local step=$1
    local merged_dir="${MERGED_MODEL_BASE}/${MODEL_IDENTIFIER}_step${step}"
    local result_dir="${RESULT_BASE}/${MODEL_IDENTIFIER}_step${step}"
    local result_file="${result_dir}/eval_${EVAL_LENGTH}_${EVAL_API}.jsonl"
    local eval_model_name="${EVAL_MODEL_NAME:-${SERVED_MODEL_NAME:-$(basename "${merged_dir}")}}"

    if [ ! -d "${merged_dir}" ]; then
        echo "Error: Merged model not found: ${merged_dir}"
        return 1
    fi

    mkdir -p "${result_dir}"

    if [ -f "${result_file}" ] && [ "${FORCE_EVAL}" != "yes" ]; then
        log_step "Evaluation results already exist: ${result_file}"
        return 0
    fi

    log_step "Running evaluation for global_step_${step}"

    cd "${MEMAGENT_ROOT}/taskutils/memory_eval"

    local force_flag=""
    if [ "${FORCE_EVAL}" == "yes" ]; then
        force_flag="--force"
    fi

    python ruler_hqa.py \
        --length "${EVAL_LENGTH}" \
        --model "${eval_model_name}" \
        --tokenizer "${merged_dir}" \
        --save_dir "${result_dir}" \
        --save_file "eval_${EVAL_LENGTH}_${EVAL_API}" \
        --api "${EVAL_API}" \
        --n_proc "${EVAL_N_PROC}" \
        --url "${URL}" \
        --api_key "${API_KEY}" \
        ${force_flag} \
        2>&1 | tee "${LOG_DIR}/eval_step${step}.log"

    cd "${MEMAGENT_ROOT}"

    log_step "Evaluation completed: ${result_file}"
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

function cleanup_merged_models() {
    if [ "${KEEP_MERGED_MODELS}" != "yes" ]; then
        log_section "Cleaning up merged models"
        for step in ${CHECKPOINT_STEPS}; do
            local merged_dir="${MERGED_MODEL_BASE}/${MODEL_IDENTIFIER}_step${step}"
            if [ -d "${merged_dir}" ]; then
                log_step "Removing: ${merged_dir}"
                rm -rf "${merged_dir}"
            fi
        done
    fi
}

function print_summary() {
    log_section "Evaluation Summary"

    local summary_file="${RESULT_BASE}/summary_${MODEL_IDENTIFIER}_eval_${EVAL_LENGTH}_${EVAL_API}.json"

    echo "Checkpoint  |  F1    |  EM    |  Sub-EM"
    echo "------------|--------|--------|--------"

    for step in ${CHECKPOINT_STEPS}; do
        local result_file="${RESULT_BASE}/${MODEL_IDENTIFIER}_step${step}/eval_${EVAL_LENGTH}_${EVAL_API}.jsonl"

        if [ -f "${result_file}" ]; then
            local stats=$(python3 -c "
import json
scores = {'f1': [], 'em': [], 'sub_em': []}
try:
    with open('${result_file}') as f:
        for line in f:
            item = json.loads(line)
            scores['f1'].append(item.get('judge_f1', 0))
            scores['em'].append(item.get('judge_em', 0))
            scores['sub_em'].append(item.get('judge_sub_em', 0))
    f1 = sum(scores['f1'])*100/len(scores['f1']) if scores['f1'] else 0
    em = sum(scores['em'])*100/len(scores['em']) if scores['em'] else 0
    sub_em = sum(scores['sub_em'])*100/len(scores['sub_em']) if scores['sub_em'] else 0
    print(f'{f1:.2f} | {em:.2f} | {sub_em:.2f}')
except Exception as e:
    print('Error | Error | Error')
" 2>/dev/null)
            printf "%-11s | %s\n" "step_${step}" "${stats}"
        else
            printf "%-11s | N/A\n" "step_${step}"
        fi
    done

    SUMMARY_FILE="${summary_file}" \
    RESULT_BASE_PATH="${RESULT_BASE}" \
    EVAL_LENGTH_VALUE="${EVAL_LENGTH}" \
    EVAL_API_VALUE="${EVAL_API}" \
    CHECKPOINT_STEPS_VALUE="${CHECKPOINT_STEPS}" \
    MODEL_IDENTIFIER="${MODEL_IDENTIFIER}" \
    python3 - <<'PY'
import json
import os
from pathlib import Path

steps = [step for step in os.environ.get("CHECKPOINT_STEPS_VALUE", "").split() if step]
result_base = os.environ.get("RESULT_BASE_PATH", "")
eval_length = os.environ.get("EVAL_LENGTH_VALUE", "")
eval_api = os.environ.get("EVAL_API_VALUE", "")
summary_file = os.environ.get("SUMMARY_FILE", "")

summary = []
model_id = os.environ.get("MODEL_IDENTIFIER", "model")
for step in steps:
    result_path = Path(result_base) / f"{model_id}_step{step}" / f"eval_{eval_length}_{eval_api}.jsonl"
    entry = {
        "checkpoint": f"step_{step}",
        "result_file": str(result_path),
        "f1": None,
        "em": None,
        "sub_em": None,
        "status": "missing"
    }
    if result_path.is_file():
        scores = {"f1": [], "em": [], "sub_em": []}
        try:
            with result_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    scores["f1"].append(item.get("judge_f1", 0))
                    scores["em"].append(item.get("judge_em", 0))
                    scores["sub_em"].append(item.get("judge_sub_em", 0))

            if scores["f1"]:
                entry["f1"] = round(sum(scores["f1"]) * 100 / len(scores["f1"]), 2)
                entry["em"] = round(sum(scores["em"]) * 100 / len(scores["em"]), 2)
                entry["sub_em"] = round(sum(scores["sub_em"]) * 100 / len(scores["sub_em"]), 2)
                entry["status"] = "ok"
        except Exception:
            entry["status"] = "error"
    summary.append(entry)

output_path = Path(summary_file)
output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open('w') as f:
    json.dump(summary, f, indent=2)
PY

    echo ""
    echo "Summary JSON saved to: ${summary_file}"
    echo "Detailed results saved in: ${RESULT_BASE}"
    echo "Logs saved in: ${LOG_DIR}"
}

# ==================== Main Execution ====================
log_section "MemAgent Checkpoint Evaluation"
log_step "Mode: ${MODE}"
log_step "Model Identifier: ${MODEL_IDENTIFIER}"
log_step "Checkpoints: ${CHECKPOINT_STEPS}"
log_step "Evaluation Length: ${EVAL_LENGTH}"
log_step "GPU: ${CUDA_VISIBLE_DEVICES}"

# Trap to ensure cleanup on exit
trap 'stop_vllm_service' EXIT

# Process each checkpoint
for step in ${CHECKPOINT_STEPS}; do
    log_section "Processing Checkpoint: global_step_${step}"

    merged_dir="${MERGED_MODEL_BASE}/${MODEL_IDENTIFIER}_step${step}"

    # Step 1: Merge checkpoint
    if [ "${MODE}" != "eval-only" ]; then
        if ! merge_checkpoint "${step}"; then
            echo "Skipping step ${step} due to merge failure"
            continue
        fi
    else
        if [ ! -d "${merged_dir}" ]; then
            echo "Error: Merged model not found for step ${step}: ${merged_dir}"
            continue
        fi
    fi

    # Skip evaluation if merge-only mode
    if [ "${MODE}" == "merge-only" ]; then
        continue
    fi

    # Step 2: Start vLLM service
    if ! start_vllm_service "${merged_dir}" "${step}"; then
        echo "Skipping evaluation for step ${step} due to vLLM startup failure"
        continue
    fi

    # Step 3: Run evaluation
    if ! run_evaluation "${step}"; then
        echo "Warning: Evaluation failed for step ${step}"
    fi

    # Stop vLLM for next iteration
    stop_vllm_service

    echo ""
done

# Cleanup
if [ "${MODE}" != "merge-only" ]; then
    cleanup_merged_models
fi

# Print summary
if [ "${MODE}" != "merge-only" ]; then
    print_summary
fi

log_section "Evaluation Complete!"
