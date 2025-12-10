#!/bin/bash
set -x

# 7B LoRA Training Script - Optimized for 2 GPUs (or even 1 GPU)
# LoRA dramatically reduces memory usage, making 7B training feasible on fewer GPUs

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate memagent

# Ensure not connected to existing Ray cluster
export RAY_ADDRESS=""
unset RAY_ADDRESS 2>/dev/null || true
export RAY_TMPDIR=/home/admin123/dl/MemAgent/outputs/ray_tmp_lora
mkdir -p "$RAY_TMPDIR"

# Use 2 GPUs for LoRA training (GPUs 6,7 are free)
# With LoRA, 2x 48GB GPUs can handle 7B model comfortably
export CUDA_VISIBLE_DEVICES=6,7
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# NCCL optimization for multi-GPU
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=SYS
export NCCL_SHM_DISABLE=1
export NCCL_IGNORE_DISABLED_P2P=1
export NCCL_SOCKET_IFNAME=ens111f0
export GLOO_SOCKET_IFNAME=ens111f0

# Memory optimization
# CRITICAL: expandable_segments is incompatible with vLLM v1 memory pool
# Must explicitly unset this variable to avoid conflicts
unset PYTORCH_CUDA_ALLOC_CONF 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF=""

# Distributed configuration
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-2}  # Using 2 GPUs (6,7)
FSDP_SIZE=${FSDP_SIZE:-$((NNODES * NGPUS_PER_NODE))}

PROJ_ROOT=/home/admin123/dl/MemAgent/outputs
DATASET_ROOT=/home/admin123/dl/MemAgent/taskutils/memory_data

MODEL_PATH=/mnt/ssd2/models/Qwen2.5-7B-Instruct
VAL_PATH="${DATASET_ROOT}/hotpotqa/hotpotqa_dev_20.parquet"
TRAIN_PATH="${DATASET_ROOT}/hotpotqa/hotpotqa_train_1k.parquet"
EXP=lora_1k_2gpu_r64
PROJ_DIR=${PROJ_ROOT}/${EXP}

# vLLM inference length
MAXLEN=256
MAX_NEW_TOKEN=64

# LoRA allows much higher learning rate than full fine-tuning!
# Full fine-tuning uses 1e-6, LoRA can use 50x higher
LEARNING_RATE=5e-5

# Token limit per GPU - optimized for 2x 48GB GPUs
# With more memory per GPU (2 GPUs vs 4), we can increase this
MAX_TOKEN_PER_GPU=16000  # Increased for 2-GPU setup with 48GB each

# Create train data if not exists
if [ ! -f "$TRAIN_PATH" ]; then
    echo "1k training data file not found, creating..."
    python3 /home/admin123/dl/MemAgent/scripts/create_train_1k.py \
        --input "${DATASET_ROOT}/hotpotqa/hotpotqa_train.parquet" \
        --output "$TRAIN_PATH" \
        --num_samples 1000
fi

# Disconnect from any existing Ray session
python3 -c "import ray; ray.shutdown()" 2>/dev/null || true
export RAY_DISABLE_IMPORT_WARNING=1

echo "========================================"
echo "Starting LoRA PPO Training (2 GPU Setup)"
echo "========================================"
echo "Experiment: ${EXP}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES} (${NGPUS_PER_NODE} x 48GB GPUs)"
echo "Learning Rate: ${LEARNING_RATE} (50x higher than full fine-tuning!)"
echo "LoRA Rank: 64"
echo "LoRA Alpha: 32"
echo "Target Modules: all-linear"
echo "Max Token Per GPU: ${MAX_TOKEN_PER_GPU}"
echo "vLLM GPU Memory: 40% (optimized for 2-GPU 48GB setup)"
echo "Expected Peak Memory: ~32-35GB per GPU"
echo "Expected: ~50% memory reduction vs full fine-tuning"
echo "========================================"

python3 -m verl.trainer.main_ppo \
    recurrent.enable=memory \
    recurrent.memory.config.chunk_size=1536 \
    recurrent.memory.config.max_chunks=16 \
    recurrent.memory.config.max_memorization_length=${MAX_NEW_TOKEN} \
    recurrent.memory.config.max_final_response_length=${MAX_NEW_TOKEN} \
    algorithm.adv_estimator=grpo \
    algorithm.grpo_use_adv=False \
    trainer.save_freq=500 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    trainer.logger=['console'] \
    actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
    actor_rollout_ref.actor.clip_ratio_high=0.20 \
    actor_rollout_ref.actor.entropy_coeff=0.000 \
    data.train_files=$TRAIN_PATH \
    data.val_files=$VAL_PATH \
    data.shuffle=False \
    data.filter_overlong_prompts=True \
    data.train_batch_size=8 \
    data.truncation='center' \
    +data.context_key='context' \
    data.max_prompt_length=$MAXLEN \
    data.max_response_length=$MAX_NEW_TOKEN \
    reward_model.reward_manager='naive' \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    actor_rollout_ref.actor.optim.lr=${LEARNING_RATE} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKEN_PER_GPU} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${MAX_TOKEN_PER_GPU} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAX_TOKEN_PER_GPU} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=none \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.load_format=dummy_dtensor \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${FSDP_SIZE} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${NGPUS_PER_NODE} \
    actor_rollout_ref.rollout.max_model_len=1024 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.40 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.critic_warmup=0 \
    critic.model.enable_gradient_checkpointing=True \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref.model.lora_rank=64 \
    +actor_rollout_ref.model.lora_alpha=32 \
    +actor_rollout_ref.model.lora_dropout=0.0 \
    +actor_rollout_ref.model.target_modules=all-linear \
    trainer.project_name='verl-hongli' \
    trainer.experiment_name=${EXP} \
    trainer.val_before_train=False \
    trainer.resume_mode=auto \
    trainer.resume_from_path=null \
    trainer.n_gpus_per_node=$NGPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.test_freq=20 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$PROJ_DIR \
    trainer.total_epochs=10

echo "========================================"
echo "Training Complete!"
echo "Results saved to: ${PROJ_DIR}"
echo "========================================"
