#!/bin/bash
set -x

# 评估检查点的脚本
# 使用方法: bash eval_checkpoint.sh <checkpoint_path>

CHECKPOINT_PATH=${1:-"/home/admin123/dl/MemAgent/outputs/memory_agent/7B_smoke/global_step_12"}
OUTPUT_DIR=${2:-"/home/admin123/dl/MemAgent/outputs/memory_agent/7B_smoke/eval_results"}

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 检查GPU显存使用情况
echo "检查GPU显存使用情况..."
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | while IFS=',' read -r idx used total; do
    used_pct=$((used * 100 / total))
    echo "GPU $idx: ${used}MB / ${total}MB (${used_pct}%)"
    if [ $used_pct -gt 80 ]; then
        echo "警告: GPU $idx 显存使用率超过80%，可能导致OOM错误"
        echo "建议: 先停止其他占用显存的进程，或使用较小的批次大小"
    fi
done

NNODES=1
NGPUS_PER_NODE=1
PROJ_ROOT=/home/admin123/dl/MemAgent/outputs
DATASET_ROOT=/home/admin123/dl/MemAgent/taskutils/memory_data

MODEL_PATH=/mnt/ssd2/models/Qwen2.5-0.5B-Instruct
VAL_PATH="${DATASET_ROOT}/hotpotqa/hotpotqa_dev_20.parquet"
TRAIN_PATH="${DATASET_ROOT}/hotpotqa/hotpotqa_train_100.parquet"
EXP=memory_agent/7B_smoke
PROJ_DIR=${PROJ_ROOT}/${EXP}

MAXLEN=4096
MAX_NEW_TOKEN=512

# 只运行验证，不训练
python3 -m verl.trainer.main_ppo \
    recurrent.enable=memory \
    recurrent.memory.config.chunk_size=5000 \
    algorithm.adv_estimator=grpo \
    algorithm.grpo_use_adv=False \
    trainer.save_freq=999 \
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
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=1 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.project_name='verl-hongli' \
    trainer.experiment_name=${EXP} \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$NGPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.test_freq=5 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$PROJ_DIR \
    trainer.total_epochs=0 \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path=$CHECKPOINT_PATH \
    trainer.validation_data_dir=$OUTPUT_DIR

echo ""
echo "=========================================="
echo "验证结果将保存到: $OUTPUT_DIR"
echo "=========================================="
echo ""
echo "结果文件说明："
echo "  1. <step>.jsonl - 每个样本的详细结果（JSONL格式）"
echo "     每行一个JSON对象，包含: input, output, score, step 等字段"
echo ""
echo "  2. 验证完成后会自动生成格式化的JSON文件"
echo ""
echo "提示: 验证完成后，结果文件会自动保存在 $OUTPUT_DIR 目录"
echo ""

