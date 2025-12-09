#!/bin/bash
set -x

# 7B 多卡训练脚本（默认使用 4 卡 FSDP 配置，固定使用 GPU 1、2、6、7；如需更改请直接修改本文件）

# 激活conda环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate memagent

# 确保不连接到现有 Ray 集群
export RAY_ADDRESS=""
unset RAY_ADDRESS 2>/dev/null || true
# 为当前作业单独指定 Ray 临时目录，避免影响其他会话
export RAY_TMPDIR=/home/admin123/dl/MemAgent/outputs/ray_tmp
mkdir -p "$RAY_TMPDIR"

# 固定使用 4 GPU（Text Memory mode 4卡 FSDP，优化显存分配）；如需改卡，直接修改这里
export CUDA_VISIBLE_DEVICES=1,2,3,4
# 按 PCI 拓扑排序设备，避免映射混乱
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# 关闭 NCCL P2P/IB，防止不支持的 peer access 报错
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=SYS
export NCCL_SHM_DISABLE=1
export NCCL_IGNORE_DISABLED_P2P=1
# 锁定物理网卡，避免走 docker/lo 回环
export NCCL_SOCKET_IFNAME=ens111f0
export GLOO_SOCKET_IFNAME=ens111f0
# vLLM 0.7.3+ 需要禁用 expandable_segments（与 CuMem allocator 冲突）
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 分布式配置，4 GPU FSDP mode
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
FSDP_SIZE=${FSDP_SIZE:-$((NNODES * NGPUS_PER_NODE))}

PROJ_ROOT=/home/admin123/dl/MemAgent/outputs
DATASET_ROOT=/home/admin123/dl/MemAgent/taskutils/memory_data

MODEL_PATH=/mnt/ssd2/models/Qwen2.5-7B-Instruct
VAL_PATH="${DATASET_ROOT}/hotpotqa/hotpotqa_dev_20.parquet"
TRAIN_PATH="${DATASET_ROOT}/hotpotqa/hotpotqa_train_1k.parquet"
EXP=text_1k_4gpu_optimized
PROJ_DIR=${PROJ_ROOT}/${EXP}

# vLLM 推理长度；Recurrent 框架使用 task config 的 max_length
MAXLEN=256
MAX_NEW_TOKEN=64

# 多卡FSDP优化：每卡显存更充裕，可以增大token限制
# 原配置：2656 tokens/GPU (2卡)
# v2配置：4000 tokens/GPU (3卡, max_chunks=8) - 48% 覆盖率，无法进入FINAL TURN
# v3配置：8000 tokens/GPU (3卡, max_chunks=16) - 95% 覆盖率，可进入FINAL TURN
MAX_TOKEN_PER_GPU=8000

# 如果1k数据文件不存在，先创建它
if [ ! -f "$TRAIN_PATH" ]; then
    echo "1k训练数据文件不存在，正在创建..."
    python3 /home/admin123/dl/MemAgent/scripts/create_train_1k.py \
        --input "${DATASET_ROOT}/hotpotqa/hotpotqa_train_1k.parquet" \
        --output "$TRAIN_PATH" \
        --num_samples 1000
fi

# 断开当前进程的 Ray 连接，强制新建本地实例
python3 -c "import ray; ray.shutdown()" 2>/dev/null || true
export RAY_DISABLE_IMPORT_WARNING=1

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
    actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
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
    actor_rollout_ref.model.path=$MODEL_PATH  \
    actor_rollout_ref.actor.optim.lr=1e-6 \
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
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.critic_warmup=0 \
    critic.model.enable_gradient_checkpointing=True \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=False \
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
