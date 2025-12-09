#!/bin/bash
# Environment Check Script for 7B 4-GPU Model Evaluation
# Run this before evaluation to verify configuration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/eval_config_7B_4gpu.rc"

echo "========================================="
echo "7B 4-GPU Model Evaluation - Pre-flight Check"
echo "========================================="
echo ""

# Load configuration
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "❌ Error: Configuration file not found: ${CONFIG_FILE}"
    exit 1
fi
echo "✓ Configuration file found: ${CONFIG_FILE}"
source "${CONFIG_FILE}"

# Check conda environment
echo ""
echo "Checking Conda environment..."
if [ -f "${CONDA_SH}" ]; then
    source "${CONDA_SH}"
    if conda activate "${CONDA_ENV}" 2>/dev/null; then
        echo "✓ Conda environment '${CONDA_ENV}' activated"
        python_version=$(python --version 2>&1)
        echo "  Python: ${python_version}"
    else
        echo "❌ Failed to activate conda environment: ${CONDA_ENV}"
        exit 1
    fi
else
    echo "⚠️  Warning: Conda not found at ${CONDA_SH}"
    echo "  Assuming environment is already active"
fi

# Check Python packages
echo ""
echo "Checking Python packages..."
required_packages=("torch" "transformers" "vllm" "safetensors")
all_packages_ok=true
for pkg in "${required_packages[@]}"; do
    if python -c "import ${pkg}" 2>/dev/null; then
        version=$(python -c "import ${pkg}; print(getattr(${pkg}, '__version__', 'unknown'))" 2>/dev/null)
        echo "  ✓ ${pkg} (${version})"
    else
        echo "  ❌ ${pkg} not found"
        all_packages_ok=false
    fi
done

if [ "${all_packages_ok}" = false ]; then
    echo ""
    echo "❌ Some required packages are missing. Please install them first."
    exit 1
fi

# Check GPU availability
echo ""
echo "Checking GPU configuration..."
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    echo ""
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits | while read line; do
        gpu_id=$(echo "$line" | cut -d',' -f1 | tr -d ' ')
        if [[ "${CUDA_VISIBLE_DEVICES}" == *"${gpu_id}"* ]]; then
            echo "  ✓ GPU ${line}"
        fi
    done

    # Check if evaluation GPU has enough memory
    eval_gpu=$(echo "${CUDA_VISIBLE_DEVICES}" | cut -d',' -f1)
    free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${eval_gpu}" 2>/dev/null || echo "0")
    if [ "${free_mem}" -gt 10000 ]; then
        echo "  ✓ GPU ${eval_gpu} has ${free_mem} MiB free (sufficient for 7B model)"
    else
        echo "  ⚠️  GPU ${eval_gpu} only has ${free_mem} MiB free"
        echo "     7B model typically needs 15-20 GB. Consider freeing up memory."
    fi
else
    echo "  ⚠️  nvidia-smi not found. Cannot verify GPU status."
fi

# Check base model
echo ""
echo "Checking base model..."
if [ -d "${BASE_MODEL}" ]; then
    echo "✓ Base model found: ${BASE_MODEL}"
    if [ -f "${BASE_MODEL}/config.json" ]; then
        model_type=$(python -c "import json; print(json.load(open('${BASE_MODEL}/config.json')).get('model_type', 'unknown'))" 2>/dev/null)
        echo "  Model type: ${model_type}"
    fi
else
    echo "❌ Base model not found: ${BASE_MODEL}"
    exit 1
fi

# Check checkpoint directory
echo ""
echo "Checking checkpoint directory..."
if [ -d "${CHECKPOINT_BASE}" ]; then
    echo "✓ Checkpoint base found: ${CHECKPOINT_BASE}"
    echo "  Available checkpoints:"
    for step in ${CHECKPOINT_STEPS}; do
        ckpt_dir="${CHECKPOINT_BASE}/global_step_${step}/actor"
        if [ -d "${ckpt_dir}" ]; then
            # Count FSDP shards
            shard_count=$(ls -1 "${ckpt_dir}"/model_world_size_*_rank_*.pt 2>/dev/null | wc -l)
            if [ "${shard_count}" -eq 4 ]; then
                echo "    ✓ global_step_${step} (4 FSDP shards found)"
            else
                echo "    ⚠️  global_step_${step} (expected 4 shards, found ${shard_count})"
            fi
        else
            echo "    ❌ global_step_${step} (not found)"
        fi
    done
else
    echo "❌ Checkpoint base not found: ${CHECKPOINT_BASE}"
    exit 1
fi

# Check data directory
echo ""
echo "Checking evaluation data..."
if [ -d "${DATAROOT}" ]; then
    echo "✓ Data directory found: ${DATAROOT}"
    data_files=$(ls -1 "${DATAROOT}"/*.parquet 2>/dev/null | wc -l)
    echo "  Found ${data_files} parquet file(s)"
else
    echo "⚠️  Data directory not found: ${DATAROOT}"
    echo "  You may need to prepare evaluation data first"
fi

# Check output directories
echo ""
echo "Checking output directories..."
for dir in "${MERGED_MODEL_BASE}" "${RESULT_BASE}" "${LOG_DIR}"; do
    if [ -d "${dir}" ]; then
        echo "  ✓ ${dir}"
    else
        echo "  ℹ️  ${dir} (will be created)"
        mkdir -p "${dir}" 2>/dev/null && echo "     Created successfully" || echo "     ⚠️  Failed to create"
    fi
done

# Check port availability
echo ""
echo "Checking vLLM port availability..."
if command -v lsof >/dev/null 2>&1; then
    if lsof -i ":${VLLM_PORT}" >/dev/null 2>&1; then
        echo "  ⚠️  Port ${VLLM_PORT} is already in use"
        echo "     Process:"
        lsof -i ":${VLLM_PORT}" | tail -n +2
        echo "     The evaluation script will automatically stop the existing service"
    else
        echo "  ✓ Port ${VLLM_PORT} is available"
    fi
else
    echo "  ℹ️  lsof not available, cannot check port status"
fi

# Check training vs evaluation parameter consistency
echo ""
echo "Checking parameter consistency..."
echo "  Training script: 7Bnodes.sh"
echo "  Evaluation config: eval_config_7B_4gpu.rc"
echo ""

# Extract parameters from training script
if [ -f "${SCRIPT_DIR}/7Bnodes.sh" ]; then
    train_chunk=$(grep -E "recurrent.memory.config.chunk_size=" "${SCRIPT_DIR}/7Bnodes.sh" | sed 's/.*chunk_size=\([0-9]*\).*/\1/' | head -1)
    train_max_new=$(grep -E "max_memorization_length=" "${SCRIPT_DIR}/7Bnodes.sh" | grep -oE '\$\{[A-Z_]+\}|[0-9]+' | tail -1)
    # Resolve variable if needed
    if [[ "${train_max_new}" == \$* ]]; then
        var_name=$(echo "${train_max_new}" | sed 's/\${\(.*\)}/\1/')
        train_max_new=$(grep -E "^${var_name}=" "${SCRIPT_DIR}/7Bnodes.sh" | sed 's/.*=\([0-9]*\).*/\1/' | head -1)
    fi
    train_model=$(grep -E "^MODEL_PATH=" "${SCRIPT_DIR}/7Bnodes.sh" | sed 's/.*=\(.*\)/\1/' | head -1)

    echo "  Training parameters:"
    echo "    chunk_size: ${train_chunk}"
    echo "    max_new_tokens: ${train_max_new}"
    echo "    model: ${train_model}"
    echo ""
    echo "  Evaluation parameters:"
    echo "    RECURRENT_CHUNK_SIZE: ${RECURRENT_CHUNK_SIZE}"
    echo "    RECURRENT_MAX_NEW: ${RECURRENT_MAX_NEW}"
    echo "    BASE_MODEL: ${BASE_MODEL}"
    echo ""

    if [ "${train_chunk}" = "${RECURRENT_CHUNK_SIZE}" ] && [ "${train_max_new}" = "${RECURRENT_MAX_NEW}" ]; then
        echo "  ✓ Parameters match!"
    else
        echo "  ⚠️  Parameters mismatch detected!"
        echo "     This may cause evaluation issues."
    fi
else
    echo "  ℹ️  Training script not found, skipping consistency check"
fi

# Summary
echo ""
echo "========================================="
echo "Pre-flight Check Complete"
echo "========================================="
echo ""
echo "Configuration Summary:"
echo "  Model: 7B (${MODEL_IDENTIFIER})"
echo "  Checkpoints: ${CHECKPOINT_STEPS}"
echo "  Evaluation Length: ${EVAL_LENGTH} docs"
echo "  vLLM Port: ${VLLM_PORT}"
echo "  GPU: ${CUDA_VISIBLE_DEVICES}"
echo "  Chunk Size: ${RECURRENT_CHUNK_SIZE}"
echo "  Max New Tokens: ${RECURRENT_MAX_NEW}"
echo ""
echo "Next steps:"
echo "  1. If all checks passed, run: bash eval_7B_4gpu.sh"
echo "  2. For single checkpoint: bash eval_7B_4gpu.sh --step 2000"
echo "  3. For merge only: bash eval_7B_4gpu.sh --merge-only"
echo ""
echo "For more information, see: EVAL_7B_4GPU_README.md"
