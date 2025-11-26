#!/bin/bash
set -x

# 最小化测试脚本 - 验证KV cache基础功能

# 强制使用GPU 1
export CUDA_VISIBLE_DEVICES=1

# 激活conda环境（关键！确保Ray workers使用正确的Python环境）
source ~/anaconda3/etc/profile.d/conda.sh
conda activate memagent

# 确保不连接到现有集群（不执行ray stop，避免影响其他GPU进程）
unset RAY_ADDRESS 2>/dev/null || true
export RAY_ADDRESS=""

# 使用独立的临时目录，避免与其他Ray实例冲突
export RAY_TMPDIR="/tmp/ray_kv_test_$$"
mkdir -p "$RAY_TMPDIR"

# 添加Ray配置，强制启动新集群
export RAY_IGNORE_VERSION_MISMATCH=1

# 检查极小数据集是否存在（只取1条样本，方便快速跑通前向+反向）
TEST_DATA="/home/admin123/dl/MemAgent/taskutils/memory_data/hotpotqa/hotpotqa_dev_1.parquet"
if [ ! -f "$TEST_DATA" ]; then
    echo "警告: 测试数据不存在，尝试创建..."
    python3 << 'EOF'
import pandas as pd
import os

input_file = "/home/admin123/dl/MemAgent/taskutils/memory_data/hotpotqa/hotpotqa_dev.parquet"
output_file = "/home/admin123/dl/MemAgent/taskutils/memory_data/hotpotqa/hotpotqa_dev_1.parquet"

if not os.path.exists(input_file):
    print(f"错误: 源文件不存在: {input_file}")
    exit(1)

df = pd.read_parquet(input_file)
df_small = df.head(1)
df_small.to_parquet(output_file)
print(f"创建测试数据: {len(df_small)} 条样本")
EOF
fi

python3 test_kv_wrapper.py \
    recurrent.enable=memory \
    recurrent.memory.path=/home/admin123/dl/MemAgent/recurrent/impls/kvcache_memory.py \
    recurrent.memory.name=REGISTER \
    recurrent.memory.config.chunk_size=256 \
    recurrent.memory.config.max_chunks=1 \
    recurrent.memory.config.max_prompt_length=1536 \
    recurrent.memory.config.max_memorization_length=128 \
    recurrent.memory.config.max_final_response_length=128 \
    +recurrent.memory.config.kv_cache_max_length=2048 \
    +recurrent.memory.config.kv_cache_dtype=bfloat16 \
    +recurrent.memory.config.reuse_prefill=True \
    +recurrent.memory.config.prompt_as_first_chunk=True \
    algorithm.adv_estimator=grpo \
    algorithm.grpo_use_adv=False \
    algorithm.use_kl_in_reward=False \
    trainer.save_freq=999 \
    trainer.logger=['console'] \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=5 \
    actor_rollout_ref.actor.clip_ratio_high=0.20 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    data.train_files=/home/admin123/dl/MemAgent/taskutils/memory_data/hotpotqa/hotpotqa_dev_1.parquet \
    data.val_files=/home/admin123/dl/MemAgent/taskutils/memory_data/hotpotqa/hotpotqa_dev_1.parquet \
    data.train_batch_size=1 \
    data.val_batch_size=1 \
    data.shuffle=False \
    data.truncation='center' \
    +data.context_key='context' \
    data.max_prompt_length=1536 \
    data.max_response_length=128 \
    reward_model.reward_manager='naive' \
    actor_rollout_ref.model.path=/mnt/ssd2/models/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=none \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=1 \
    actor_rollout_ref.rollout.name=hf_kv \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=1024 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=1024 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=1024 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.project_name='verl-test' \
    trainer.experiment_name='kv_cache_test' \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.test_freq=1 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=/home/admin123/dl/MemAgent/outputs/kv_test \
    trainer.total_epochs=1

echo "===== 测试完成 ====="
echo "检查输出日志中是否有："
echo "  1. [Prefill] 相关日志"
echo "  2. [Decode] 相关日志"
echo "  3. KV cache 长度信息"
echo "  4. 最终的 validation accuracy"
