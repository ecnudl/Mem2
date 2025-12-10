#!/bin/bash
set -x

# Quick smoke test - 100 steps only to verify shaped reward works
echo "========================================="
echo "SMOKE TEST: Shaped Reward (100 steps)"
echo "========================================="

source ~/anaconda3/etc/profile.d/conda.sh
conda activate memagent

export RAY_ADDRESS=""
unset RAY_ADDRESS 2>/dev/null || true
export RAY_TMPDIR=/home/admin123/dl/MemAgent/outputs/ray_tmp_lora_smoke
mkdir -p "$RAY_TMPDIR"

export CUDA_VISIBLE_DEVICES=6,7
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
export USE_SHAPED_REWARD=true

NNODES=1
NGPUS_PER_NODE=2
FSDP_SIZE=2

PROJ_ROOT=/home/admin123/dl/MemAgent/outputs
DATASET_ROOT=/home/admin123/dl/MemAgent/taskutils/memory_data

MODEL_PATH=/mnt/ssd2/models/Qwen2.5-7B-Instruct
VAL_PATH="${DATASET_ROOT}/hotpotqa/hotpotqa_dev_20.parquet"
TRAIN_PATH="${DATASET_ROOT}/hotpotqa/hotpotqa_train_1k.parquet"
EXP=smoke_test_shaped_reward
PROJ_DIR=${PROJ_ROOT}/${EXP}

MAXLEN=256
MAX_NEW_TOKEN=64
LEARNING_RATE=1e-5
MAX_TOKEN_PER_GPU=16000

python3 -c "import ray; ray.shutdown()" 2>/dev/null || true
export RAY_DISABLE_IMPORT_WARNING=1

echo "Starting smoke test with shaped reward..."
echo "Will train for ~100 steps (1 epoch with small batch)"

python3 -m verl.trainer.main_ppo \
    recurrent.enable=memory \
    recurrent.memory.config.chunk_size=1536 \
    recurrent.memory.config.max_chunks=16 \
    recurrent.memory.config.max_memorization_length=${MAX_NEW_TOKEN} \
    recurrent.memory.config.max_final_response_length=${MAX_NEW_TOKEN} \
    algorithm.adv_estimator=grpo \
    algorithm.grpo_use_adv=False \
    trainer.save_freq=1000 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    trainer.logger=['console'] \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.clip_ratio_high=0.20 \
    actor_rollout_ref.actor.entropy_coeff=0.000 \
    data.train_files=$TRAIN_PATH \
    data.val_files=$VAL_PATH \
    data.shuffle=False \
    data.filter_overlong_prompts=True \
    data.train_batch_size=4 \
    data.truncation='center' \
    +data.context_key='context' \
    data.max_prompt_length=$MAXLEN \
    data.max_response_length=$MAX_NEW_TOKEN \
    reward_model.reward_manager='naive' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=${LEARNING_RATE} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
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
    +actor_rollout_ref.model.lora_rank=32 \
    +actor_rollout_ref.model.lora_alpha=16 \
    +actor_rollout_ref.model.lora_dropout=0.05 \
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
    trainer.total_epochs=1

echo "========================================="
echo "Smoke test complete!"
echo "Check: outputs/${EXP}/"
echo "========================================="
