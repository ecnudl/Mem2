#!/bin/bash
# MemAgent SFT + RL Training Pipeline
# This script demonstrates the full training workflow: Base Model → SFT → RL

set -e

# ============================================
# Configuration
# ============================================

# Paths
export BASE_MODEL=/path/to/Qwen2.5-7B-Instruct-128K
export SFT_DATA_ROOT=/path/to/sft_data
export RL_DATA_ROOT=/path/to/rl_data
export OUTPUT_ROOT=/path/to/outputs

# Training settings
export NUM_GPUS=4
export SFT_EPOCHS=3
export RL_STEPS=5000

# ============================================
# Stage 1: Supervised Fine-Tuning (SFT)
# ============================================

echo "=========================================="
echo "Stage 1: Running SFT Training"
echo "=========================================="

export SFT_OUTPUT=$OUTPUT_ROOT/sft_checkpoints

torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$SFT_DATA_ROOT/train.parquet \
    data.val_files=$SFT_DATA_ROOT/val.parquet \
    data.prompt_key=prompt \
    data.response_key=response \
    data.max_length=4096 \
    data.train_batch_size=128 \
    data.micro_batch_size_per_gpu=2 \
    model.partial_pretrain=$BASE_MODEL \
    model.enable_gradient_checkpointing=True \
    optim.lr=2e-5 \
    optim.warmup_steps_ratio=0.05 \
    optim.clip_grad=1.0 \
    optim.lr_scheduler=cosine \
    trainer.total_epochs=$SFT_EPOCHS \
    trainer.default_local_dir=$SFT_OUTPUT \
    trainer.logger=['console'] \
    trainer.project_name=memagent-sft \
    trainer.experiment_name=qwen7b-sft-stage1

# Find the best checkpoint (latest by default)
SFT_CHECKPOINT=$(ls -td $SFT_OUTPUT/global_step_* | head -1)
echo "SFT completed. Best checkpoint: $SFT_CHECKPOINT"

# ============================================
# Stage 2: Reinforcement Learning (RL/PPO)
# ============================================

echo "=========================================="
echo "Stage 2: Running RL Training (PPO)"
echo "=========================================="

export RL_OUTPUT=$OUTPUT_ROOT/rl_checkpoints
export PROJ_ROOT=$(pwd)

# Disconnect from any existing Ray cluster
export RAY_ADDRESS=""
unset RAY_ADDRESS
python3 -c "import ray; ray.shutdown()" 2>/dev/null || true

# Ray temporary directory (avoid conflicts)
export RAY_TMPDIR=/tmp/ray_$(date +%s)_$$
mkdir -p $RAY_TMPDIR

# NCCL settings for multi-GPU
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run PPO training with SFT checkpoint as initialization
python3 -m verl.trainer.main_ppo \
    data.train_files=$RL_DATA_ROOT/hotpotqa/hotpotqa_train.parquet \
    data.val_files=$RL_DATA_ROOT/hotpotqa/hotpotqa_dev.parquet \
    data.train_batch_size=512 \
    data.val_batch_size=1312 \
    data.chunk_size=1024 \
    actor_rollout_ref.model.path=$SFT_CHECKPOINT \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    critic.optim.lr=1e-5 \
    critic.model.path=$SFT_CHECKPOINT \
    critic.ppo_micro_batch_size=64 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.total_training_steps=$RL_STEPS \
    trainer.logger=['console'] \
    trainer.project_name=memagent-rl \
    trainer.experiment_name=qwen7b-rl-from-sft \
    trainer.default_hdfs_dir=$RL_OUTPUT \
    recurrent.enable=memory \
    recurrent.memory.config.chunk_size=3000 \
    recurrent.memory.config.max_new=200 \
    +recurrent.memory.config.num_chunks=null

echo "=========================================="
echo "Training Pipeline Complete!"
echo "=========================================="
echo "SFT Checkpoint: $SFT_CHECKPOINT"
echo "RL Checkpoints: $RL_OUTPUT"
echo "=========================================="

# Cleanup
rm -rf $RAY_TMPDIR
