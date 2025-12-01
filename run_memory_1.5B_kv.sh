#!/bin/bash
set -x

NNODES=1
NGPUS_PER_NODE=1
PROJ_ROOT=/home/admin123/dl/MemAgent/outputs
DATASET_ROOT=/home/admin123/dl/MemAgent/taskutils/memory_data

MODEL_PATH=/mnt/ssd2/models/Qwen2.5-1.5B-Instruct
VAL_PATH="${DATASET_ROOT}/hotpotqa/hotpotqa_dev_20.parquet"
TRAIN_PATH="${DATASET_ROOT}/hotpotqa/hotpotqa_train_1k.parquet"
EXP=memory_agent/1.5B_kv
PROJ_DIR=${PROJ_ROOT}/${EXP}

MAXLEN=1792
MAX_NEW_TOKEN=192
KV_CACHE_MAX_LEN=2560
MAX_TOKEN_LEN_PER_GPU=3584

SELECTED_GPU=2
echo "固定使用GPU ${SELECTED_GPU} 进行训练"

# 激活conda环境（关键！确保Ray workers使用正确的Python环境）
source ~/anaconda3/etc/profile.d/conda.sh
conda activate memagent

export CUDA_VISIBLE_DEVICES=${SELECTED_GPU}
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 使用独立的临时目录，避免与其他Ray实例冲突
export RAY_TMPDIR="/tmp/ray_kv_training_$$"
mkdir -p "$RAY_TMPDIR"

# 确保不连接到现有集群（不执行ray stop，避免影响其他GPU进程）
unset RAY_ADDRESS 2>/dev/null || true
export RAY_ADDRESS=""

# 添加Ray配置，强制启动新集群
export RAY_IGNORE_VERSION_MISMATCH=1

python3 -m verl.trainer.main_ppo_kv \
    recurrent.enable=memory \
    recurrent.memory.path=/home/admin123/dl/MemAgent/recurrent/impls/kvcache_memory.py \
    recurrent.memory.name=REGISTER \
    recurrent.memory.config.chunk_size=1024 \
    recurrent.memory.config.max_prompt_length=${MAXLEN} \
    recurrent.memory.config.max_memorization_length=${MAX_NEW_TOKEN} \
    recurrent.memory.config.max_final_response_length=${MAX_NEW_TOKEN} \
    recurrent.memory.config.max_chunks=1 \
    +recurrent.memory.config.kv_cache_max_length=${KV_CACHE_MAX_LEN} \
    +recurrent.memory.config.kv_cache_dtype=bfloat16 \
    +recurrent.memory.config.reuse_prefill=True \
    +recurrent.memory.config.prompt_as_first_chunk=True \
    algorithm.adv_estimator=grpo \
    algorithm.grpo_use_adv=False \
    trainer.save_freq=1000 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    trainer.logger=['console'] \
    actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
    actor_rollout_ref.actor.clip_ratio_high=0.20 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    data.train_files=${TRAIN_PATH} \
    data.val_files=${VAL_PATH} \
    data.shuffle=False \
    data.filter_overlong_prompts=True \
    data.train_batch_size=1 \
    data.truncation='center' \
    +data.context_key='context' \
    +data.default_data_source='hotpotqa' \
    data.max_prompt_length=${MAXLEN} \
    data.max_response_length=${MAX_NEW_TOKEN} \
    reward_model.reward_manager='thread' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=none \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${NGPUS_PER_NODE} \
    actor_rollout_ref.rollout.name=hf_kv \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.prompt_length=${MAXLEN} \
    actor_rollout_ref.rollout.response_length=${MAX_NEW_TOKEN} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.ppo_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
    critic.forward_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
    reward_model.forward_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.critic_warmup=0 \
    trainer.project_name='verl-hongli' \
    trainer.experiment_name=${EXP} \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.test_freq=5 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=${PROJ_DIR} \
    trainer.resume_mode=disable \
    trainer.total_epochs=10
