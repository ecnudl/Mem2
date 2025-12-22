#!/bin/bash
set -x

# 7B LoRA Training Script - FIXED Balanced Configuration for 2×48GB GPUs
# Fixed OOM issues from original balanced config
# Expected memory usage: ~28-32GB per GPU

source ~/anaconda3/etc/profile.d/conda.sh
conda activate memagent

export RAY_ADDRESS=""
unset RAY_ADDRESS 2>/dev/null || true
export RAY_TMPDIR=/tmp/ray_m4_fixed
export TMPDIR="$RAY_TMPDIR"
export TEMP="$RAY_TMPDIR"
export TMP="$RAY_TMPDIR"
mkdir -p "$RAY_TMPDIR"

export CUDA_VISIBLE_DEVICES=2,3
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Disable wandb online sync to avoid network errors
export WANDB_MODE=offline

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=SYS
export NCCL_SHM_DISABLE=1
export NCCL_IGNORE_DISABLED_P2P=1
export NCCL_SOCKET_IFNAME=ens111f0
export GLOO_SOCKET_IFNAME=ens111f0

unset PYTORCH_CUDA_ALLOC_CONF 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF=""

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-2}
FSDP_SIZE=${FSDP_SIZE:-$((NNODES * NGPUS_PER_NODE))}

PROJ_ROOT=/home/admin123/dl/MemAgent/outputs
DATASET_ROOT=/home/admin123/dl/MemAgent/taskutils/memory_data

MODEL_PATH=/mnt/ssd2/models/Qwen2.5-7B-Instruct
VAL_PATH="${DATASET_ROOT}/hotpotqa/hotpotqa_dev_100_balanced.parquet"
TRAIN_PATH="${DATASET_ROOT}/hotpotqa/hotpotqa_train_1k.parquet"
EXP=lora_2gpu_balanced_fixed_17k_n6
PROJ_DIR=${PROJ_ROOT}/${EXP}

# ============================================================================
# FIXED BALANCED CONFIGURATION
# ============================================================================

# FIX 1: Reduce context length slightly
MAXLEN=17000  # Down from 20000

# FIX 2: Reduce GRPO sampling (biggest memory saver)
ROLLOUT_N=3   # Down from 4 for 2-GPU setup
ROLLOUT_VAL_N=2  # Down from 4

# FIX 3: Reduce batch size for 2-GPU setup
TRAIN_BATCH_SIZE=6  # Down from 10 for 2-GPU
PPO_MINI_BATCH_SIZE=6

# FIX 4: Increase per-GPU token limit
MAX_TOKEN_PER_GPU=15000  # Up from 12000

# Memory agent configuration
CHUNK_SIZE=2500
MAX_CHUNKS=7  # 7 * 2500 = 17500 (covers MAXLEN)
MAX_NEW_TOKEN=1024

# Learning rates
ACTOR_LR=3e-5
CRITIC_LR=8e-5
LR_WARMUP_STEPS=100

# FIX 5: Reduce vLLM memory allocation
VLLM_GPU_MEMORY_UTIL=0.35  # Down from 0.40
VLLM_MAX_BATCHED_TOKENS=6144  # Down from 8192
VLLM_MAX_MODEL_LEN=3072  # Down from 3584, but still > CHUNK_SIZE + overhead
TENSOR_PARALLEL_SIZE=1  # Changed from 2 for 2-GPU setup

# ============================================================================

if [ ! -f "$TRAIN_PATH" ]; then
    echo "1k training data file not found, creating..."
    python3 /home/admin123/dl/MemAgent/scripts/create_train_1k.py \
        --input "${DATASET_ROOT}/hotpotqa/hotpotqa_train.parquet" \
        --output "$TRAIN_PATH" \
        --num_samples 1000
fi

python3 -c "import ray; ray.shutdown()" 2>/dev/null || true
export RAY_DISABLE_IMPORT_WARNING=1

echo "================================================================================"
echo "MemAgent PPO Training - FIXED Balanced Configuration (2×48GB GPUs)"
echo "================================================================================"
echo "Experiment: ${EXP}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo ""
echo "Configuration Summary (FIXED):"
echo "  MAXLEN: ${MAXLEN} (down from 20k)"
echo "  GRPO Sampling: n=${ROLLOUT_N} (down from 8)"
echo "  Batch Size: ${TRAIN_BATCH_SIZE} (reduced for 2-GPU)"
echo "  Effective Batch Size: ${TRAIN_BATCH_SIZE} × ${ROLLOUT_N} = $((TRAIN_BATCH_SIZE * ROLLOUT_N))"
echo "  MAX_TOKEN_PER_GPU: ${MAX_TOKEN_PER_GPU} (up from 12k)"
echo "  vLLM Memory: ${VLLM_GPU_MEMORY_UTIL} (down from 0.40)"
echo ""
echo "Expected Memory Usage: ~28-32GB per GPU (should work on 48GB GPUs)"
echo "================================================================================"
echo ""

python3 -m verl.trainer.main_ppo \
    recurrent.enable=memory \
    recurrent.memory.config.chunk_size=${CHUNK_SIZE} \
    recurrent.memory.config.max_chunks=${MAX_CHUNKS} \
    recurrent.memory.config.max_memorization_length=${MAX_NEW_TOKEN} \
    recurrent.memory.config.max_final_response_length=${MAX_NEW_TOKEN} \
    algorithm.adv_estimator=grpo \
    algorithm.grpo_use_adv=False \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.save_freq=300 \
    trainer.test_freq=15 \
    trainer.total_epochs=10 \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.val_kwargs.n=${ROLLOUT_VAL_N} \
    trainer.logger=['console'] \
    data.train_files=$TRAIN_PATH \
    data.val_files=$VAL_PATH \
    data.shuffle=False \
    data.filter_overlong_prompts=True \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.truncation='center' \
    ++data.context_key='context' \
    data.max_prompt_length=$MAXLEN \
    data.max_response_length=$MAX_NEW_TOKEN \
    reward_model.reward_manager='naive' \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    ++actor_rollout_ref.model.lora_rank=64 \
    ++actor_rollout_ref.model.lora_alpha=32 \
    ++actor_rollout_ref.model.lora_dropout=0.0 \
    ++actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKEN_PER_GPU} \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${LR_WARMUP_STEPS} \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    ++actor_rollout_ref.actor.optim.min_lr_ratio=0.1 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    ++actor_rollout_ref.actor.clip_ratio_low=0.2 \
    ++actor_rollout_ref.actor.clip_ratio_high=0.25 \
    actor_rollout_ref.actor.entropy_coeff=0.002 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=none \
    actor_rollout_ref.actor.grad_clip=1.0 \
    ++actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    ++actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${FSDP_SIZE} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${MAX_TOKEN_PER_GPU} \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TENSOR_PARALLEL_SIZE} \
    actor_rollout_ref.rollout.max_model_len=${VLLM_MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${VLLM_GPU_MEMORY_UTIL} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${VLLM_MAX_BATCHED_TOKENS} \
    ++actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.load_format=dummy_dtensor \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAX_TOKEN_PER_GPU} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    critic.optim.lr=${CRITIC_LR} \
    critic.optim.lr_warmup_steps_ratio=0.0 \
    critic.optim.warmup_style=constant \
    critic.optim.weight_decay=0.01 \
    critic.model.enable_gradient_checkpointing=True \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    critic.ppo_max_token_len_per_gpu=$((MAX_TOKEN_PER_GPU * 2)) \
    trainer.project_name='verl-memagent' \
    trainer.experiment_name=${EXP} \
    trainer.resume_mode=auto \
    trainer.resume_from_path=null \
    trainer.n_gpus_per_node=$NGPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$PROJ_DIR

echo ""
echo "================================================================================"
echo "Training Complete!"
echo "================================================================================"
