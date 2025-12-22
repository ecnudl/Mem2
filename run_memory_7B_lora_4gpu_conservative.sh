#!/bin/bash
set -x

# 7B LoRA Training Script - Conservative Configuration for 4×48GB GPUs
# Optimized for fast iteration and stability
# Expected memory usage: ~20-25GB per GPU

source ~/anaconda3/etc/profile.d/conda.sh
conda activate memagent

export RAY_ADDRESS=""
unset RAY_ADDRESS 2>/dev/null || true
export RAY_TMPDIR=/home/admin123/dl/MemAgent/outputs/ray_tmp_4gpu_conservative
mkdir -p "$RAY_TMPDIR"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_DEVICE_ORDER=PCI_BUS_ID

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=SYS
export NCCL_SHM_DISABLE=1
export NCCL_IGNORE_DISABLED_P2P=1
export NCCL_SOCKET_IFNAME=ens111f0
export GLOO_SOCKET_IFNAME=ens111f0

unset PYTORCH_CUDA_ALLOC_CONF 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF=""

NNODES=1
NGPUS_PER_NODE=4
FSDP_SIZE=4

PROJ_ROOT=/home/admin123/dl/MemAgent/outputs
DATASET_ROOT=/home/admin123/dl/MemAgent/taskutils/memory_data
MODEL_PATH=/mnt/ssd2/models/Qwen2.5-7B-Instruct
VAL_PATH="${DATASET_ROOT}/hotpotqa/hotpotqa_dev_100_balanced.parquet"
TRAIN_PATH="${DATASET_ROOT}/hotpotqa/hotpotqa_train_1k.parquet"
EXP=lora_4gpu_conservative_15k_n4
PROJ_DIR=${PROJ_ROOT}/${EXP}

# ============================================================================
# CONSERVATIVE CONFIGURATION - Fast & Stable
# ============================================================================
MAXLEN=15000
MAX_NEW_TOKEN=1024
CHUNK_SIZE=2500
MAX_CHUNKS=6  # 6 * 2500 = 15000
ROLLOUT_N=4   # Conservative GRPO sampling
ROLLOUT_VAL_N=2
TRAIN_BATCH_SIZE=16  # Larger batch for stability
PPO_MINI_BATCH_SIZE=16
MAX_TOKEN_PER_GPU=18000  # More headroom
ACTOR_LR=5e-5
CRITIC_LR=1e-4
LR_WARMUP_STEPS=50
VLLM_GPU_MEMORY_UTIL=0.45
VLLM_MAX_BATCHED_TOKENS=6144
VLLM_MAX_MODEL_LEN=2048
TENSOR_PARALLEL_SIZE=2
# ============================================================================

if [ ! -f "$TRAIN_PATH" ]; then
    python3 /home/admin123/dl/MemAgent/scripts/create_train_1k.py \
        --input "${DATASET_ROOT}/hotpotqa/hotpotqa_train.parquet" \
        --output "$TRAIN_PATH" \
        --num_samples 1000
fi

python3 -c "import ray; ray.shutdown()" 2>/dev/null || true
export RAY_DISABLE_IMPORT_WARNING=1

echo "================================================================================"
echo "MemAgent PPO Training - CONSERVATIVE Configuration"
echo "================================================================================"
echo "Use Case: Fast iteration, initial experiments, debugging"
echo "Data Coverage: 50-60% | Expected Memory: ~20-25GB/GPU | Speed: Fast"
echo "================================================================================"

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
    trainer.save_freq=500 \
    trainer.test_freq=20 \
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
    +data.context_key='context' \
    data.max_prompt_length=$MAXLEN \
    data.max_response_length=$MAX_NEW_TOKEN \
    reward_model.reward_manager='naive' \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.lora_rank=64 \
    +actor_rollout_ref.model.lora_alpha=32 \
    +actor_rollout_ref.model.lora_dropout=0.0 \
    +actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKEN_PER_GPU} \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${LR_WARMUP_STEPS} \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.grad_clip=1.0 \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
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
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.load_format=dummy_dtensor \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAX_TOKEN_PER_GPU} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    critic.optim.lr=${CRITIC_LR} \
    critic.optim.lr_warmup_steps_ratio=0.0 \
    critic.model.enable_gradient_checkpointing=True \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    critic.ppo_max_token_len_per_gpu=$((MAX_TOKEN_PER_GPU * 2)) \
    trainer.project_name='verl-memagent' \
    trainer.experiment_name=${EXP} \
    trainer.resume_mode=auto \
    trainer.n_gpus_per_node=$NGPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.default_local_dir=$PROJ_DIR

echo "Training Complete! Results: ${PROJ_DIR}"
