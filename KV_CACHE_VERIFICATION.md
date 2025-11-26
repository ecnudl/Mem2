# KV Cache 实现验证报告

## 检查结果总结 ✅

### 1. 基础设施检查

| 组件 | 状态 | 位置 | 说明 |
|------|------|------|------|
| **Rollout Backend** | ✅ 完整 | `verl/workers/rollout/naive/naive_rollout_kv.py` | 实现了 prefill/decode 两阶段 |
| **FSDP Worker** | ✅ 完整 | `verl/workers/fsdp_workers_kv.py` | 支持 `hf_kv` rollout |
| **Generation Manager** | ✅ 完整 | `recurrent/generation_manager_kv.py` | 专门处理 KV cache 流程 |
| **PPO Trainer** | ✅ 完整 | `verl/trainer/ppo/ray_trainer_kv.py` | 使用 KV generation manager |
| **Main Entry** | ✅ 完整 | `verl/trainer/main_ppo_kv.py` | 完整的启动脚本 |

### 2. 你的 kvcache_memory.py 评估

#### 优点 ✅
- 状态机设计清晰：`prefill → decode → done`
- `pending_tensors` 机制与 `generation_manager_kv.py` 完美配合
- KV cache 管理完整（concat、truncate、dtype转换）
- 配置灵活（`kv_cache_max_length`、`kv_cache_dtype`、`reuse_prefill`）

#### 改进建议（已实现在 kvcache_memory_debug.py）
- ✅ 添加详细日志跟踪 prefill/decode 流程
- ✅ 记录每个样本的 KV cache 长度变化
- ✅ 输出每个阶段的状态信息

## 快速开始

### 方式1: 运行最小测试（推荐）

```bash
cd /home/admin123/dl/MemAgent
bash test_kv_minimal.sh
```

这个脚本会：
- 使用 GPU 1
- 只用 20 条样本
- 训练 1 个 epoch
- 2 个 prefill 步骤（chunk_size=512, max_chunks=2）
- 输出详细日志

**预期输出：**
```
[Prefill Step 0] Sample 0: First chunk with prompt (prompt=5, chunk=512)
[Prefill Update] Sample 0: KV cache 0 → 517 tokens
[Prefill Step 1] Sample 0: Chunk tokens: 512 (offset=512)
[Prefill Update] Sample 0: KV cache 517 → 1029 tokens
[Prefill->Decode] All chunks processed, switching to decode phase
[Decode] Starting final generation with KV cache lengths: [1029, 1024, ...]
[Decode Update] Final answer generated
[Agent End] Total prefill steps: 2, Final KV cache lengths: [1029, 1024, ...]
```

### 方式2: 使用调试版本（查看详细信息）

修改 `test_kv_minimal.sh` 中的路径：

```bash
# 改为使用调试版本
recurrent.memory.path=/home/admin123/dl/MemAgent/recurrent/impls/kvcache_memory_debug.py \
```

### 方式3: 运行完整训练

使用你的 `run_memory_7B_kv.sh`（如果存在）或创建一个：

```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,2,3

python3 -m verl.trainer.main_ppo_kv \
    recurrent.enable=memory \
    recurrent.memory.path=/home/admin123/dl/MemAgent/recurrent/impls/kvcache_memory.py \
    recurrent.memory.name=REGISTER \
    recurrent.memory.config.chunk_size=1024 \
    recurrent.memory.config.max_chunks=3 \
    +recurrent.memory.config.kv_cache_max_length=4096 \
    +recurrent.memory.config.kv_cache_dtype=float16 \
    +recurrent.memory.config.reuse_prefill=True \
    +recurrent.memory.config.prompt_as_first_chunk=True \
    algorithm.adv_estimator=grpo \
    algorithm.grpo_use_adv=False \
    data.train_batch_size=16 \
    data.train_files=/path/to/hotpotqa_train_32k.parquet \
    actor_rollout_ref.model.path=/path/to/Qwen2.5-7B-Instruct \
    actor_rollout_ref.rollout.name=hf_kv \
    actor_rollout_ref.rollout.mode=sync \
    trainer.total_epochs=3 \
    ...
```

## 验证检查清单

运行测试后，检查以下内容：

### ✅ 阶段1: 启动检查
- [ ] Ray 成功初始化
- [ ] Worker 成功创建
- [ ] 模型加载成功
- [ ] 数据集加载成功

### ✅ 阶段2: Prefill检查
- [ ] 看到 `[Prefill Step X]` 日志
- [ ] KV cache 长度逐步增加
- [ ] 每个样本的 chunk 被正确处理
- [ ] 切换到 decode 阶段

### ✅ 阶段3: Decode检查
- [ ] 看到 `[Decode]` 日志
- [ ] KV cache 被正确传递
- [ ] 最终答案生成成功
- [ ] `final_mask` 和 `sample_index` 正确

### ✅ 阶段4: Training检查
- [ ] 奖励计算成功
- [ ] GRPO 优势计算正常
- [ ] PPO 损失计算正常
- [ ] 梯度反向传播成功
- [ ] Validation accuracy 合理（~50% 训练时，~80% 测试时）

## 常见问题排查

### 问题1: `past_key_values is None`

**原因：** Rollout backend 没有返回 KV cache

**解决：**
```bash
# 确保使用 hf_kv backend
actor_rollout_ref.rollout.name=hf_kv  # 不是 hf！
```

### 问题2: OOM (Out of Memory)

**解决方案（按优先级）：**
```bash
# 1. 减小 batch size
data.train_batch_size=8

# 2. 限制 KV cache 长度
recurrent.memory.config.kv_cache_max_length=2048

# 3. 使用 fp16
recurrent.memory.config.kv_cache_dtype=float16

# 4. 减小 chunk size
recurrent.memory.config.chunk_size=512
```

### 问题3: `RuntimeError: stage not recognized`

**原因：** 使用了错误的 generation_manager

**解决：** 确保使用 `main_ppo_kv.py` 而不是 `main_ppo.py`

```bash
python3 -m verl.trainer.main_ppo_kv ...  # ← 正确
# 不是: python3 -m verl.trainer.main_ppo ...
```

### 问题4: Validation accuracy 为 0

**可能原因：**
- `kv_cache_max_length` 过小，丢失了重要信息
- `prompt_as_first_chunk=False` 导致 prompt 不在 KV cache 中

**解决：**
```bash
recurrent.memory.config.kv_cache_max_length=8192  # 增大
recurrent.memory.config.prompt_as_first_chunk=True  # 确保开启
```

## 性能对比测试

运行以下命令对比文本模式和 KV cache 模式：

```bash
# 1. 文本模式 baseline
bash run1k.sh  # 记录: tokens/sec, GPU memory, accuracy

# 2. KV cache 模式
bash test_kv_minimal.sh  # 记录: tokens/sec, GPU memory, accuracy

# 3. 对比结果
echo "Speedup: $(bc <<< "scale=2; $KV_TOKENS_SEC / $TEXT_TOKENS_SEC")"
```

**预期结果：**
- 训练速度：KV cache 应该快 1.5-3x
- GPU 内存：KV cache 占用更多（+20-40%）
- Accuracy：应该相同或略有差异

## 调试技巧

### 1. 查看 KV cache 形状

在 `kvcache_memory.py` 的 `update()` 中添加：

```python
if cache_fragment is not None and len(cache_fragment) > 0:
    key_shape = cache_fragment[0][0].shape
    print(f"KV cache shape: {key_shape}")
    # 应该是: (batch=1, num_heads, seq_len, head_dim)
```

### 2. 验证梯度流

在 `ray_trainer_kv.py` 中添加：

```python
# 在 update_actor 之后
for name, param in model.named_parameters():
    if param.grad is None:
        print(f"Warning: {name} has no gradient!")
```

### 3. 监控 GPU 内存

```bash
watch -n 1 nvidia-smi
```

在训练过程中观察：
- Prefill 阶段：内存逐步增加
- Decode 阶段：内存略有上升（生成 response）
- PPO 更新：内存峰值

## 下一步

如果测试通过：

1. **优化超参数**
   - 调整 `chunk_size`（推荐 1024-2048）
   - 调整 `kv_cache_max_length`（根据 GPU 内存）
   - 增大 `batch_size`

2. **扩展到生产**
   - 使用完整训练数据
   - 增加 epochs
   - 启用 checkpointing
   - 使用 WandB 记录指标

3. **进一步优化**
   - 尝试 `vllm_kv` backend（如果可用）
   - 启用异步模式（需要实现）
   - 使用 FP8 量化（如果硬件支持）

## 参考

- 原始文本模式：`recurrent/impls/memory.py`
- KV cache rollout：`verl/workers/rollout/naive/naive_rollout_kv.py`
- Generation manager：`recurrent/generation_manager_kv.py`
- 配置参考：`run_memory_7B_kv.sh`
