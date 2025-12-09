# 7B 四卡训练模型评估指南

## 概述
本目录包含针对 `7Bnodes.sh` 四卡训练结果的评估配置和脚本。

## 文件说明

- **eval_config_7B_4gpu.rc**: 7B 四卡训练结果的评估配置文件
- **eval_7B_4gpu.sh**: 快速启动脚本（推荐使用）
- **run_evaluation.sh**: 通用评估脚本（底层实现）

## 配置说明

### 关键参数（已与训练配置自动对齐）

从 `7Bnodes.sh` 训练脚本中提取的参数：

```bash
RECURRENT_CHUNK_SIZE="1536"      # 与训练中的 chunk_size 匹配
RECURRENT_MAX_NEW="64"            # 与训练中的 max_memorization_length 匹配
BASE_MODEL="/mnt/ssd2/models/Qwen2.5-7B-Instruct"  # 7B 基座模型
CHECKPOINT_BASE="outputs/text_1k_4gpu_optimized"    # 训练输出目录
```

### GPU 配置

- **训练时使用**: GPU 1, 2, 3, 4 (四卡 FSDP)
- **评估时使用**: GPU 0 (单卡推理，避免冲突)
- **vLLM 端口**: 8004 (避免与其他模型端口冲突)

### Checkpoint 结构

四卡训练的 checkpoint 结构示例：
```
outputs/text_1k_4gpu_optimized/
├── global_step_1000/
│   └── actor/
│       ├── model_world_size_4_rank_0.pt
│       ├── model_world_size_4_rank_1.pt
│       ├── model_world_size_4_rank_2.pt
│       ├── model_world_size_4_rank_3.pt
│       └── ...
└── global_step_2000/
    └── actor/
        └── ...
```

## 使用方法

### 1. 基本评估（评估所有 checkpoint）

```bash
bash eval_7B_4gpu.sh
```

这将：
1. 合并 FSDP 分片（global_step_1000, global_step_2000）
2. 启动 vLLM 服务（每个 checkpoint）
3. 运行评估（默认 100 个文档）
4. 生成评估报告

### 2. 评估单个 checkpoint

```bash
bash eval_7B_4gpu.sh --step 2000
```

### 3. 只合并 checkpoint（不评估）

```bash
bash eval_7B_4gpu.sh --merge-only
```

适用场景：
- 提前准备合并好的模型
- 检查合并是否成功

### 4. 只评估（跳过合并）

```bash
bash eval_7B_4gpu.sh --eval-only
```

前提条件：
- 已经运行过 `--merge-only` 或之前已评估过
- `merged_models/7B_text_4gpu_step*/` 目录存在

### 5. 自定义文档数量

```bash
# 评估 200 个文档
bash eval_7B_4gpu.sh --length 200

# 评估 50 个文档（快速测试）
bash eval_7B_4gpu.sh --length 50 --step 2000
```

可选长度：50, 100, 200, 400, 800, 1600, 3200, 6400

### 6. 强制重新评估

编辑 `eval_config_7B_4gpu.rc`，修改：
```bash
export FORCE_MERGE="yes"  # 强制重新合并
export FORCE_EVAL="yes"   # 强制重新评估
```

## 输出结果

### 合并后的模型

```
merged_models/
├── 7B_text_4gpu_step1000/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── ...
└── 7B_text_4gpu_step2000/
    └── ...
```

### 评估结果

```
taskutils/memory_eval/results/
├── 7B_text_4gpu_step1000/
│   └── eval_100_recurrent.jsonl
├── 7B_text_4gpu_step2000/
│   └── eval_100_recurrent.jsonl
└── summary_7B_text_4gpu_eval_100_recurrent.json  # 汇总报告
```

### 日志文件

```
eval_logs_7B_4gpu/
├── merge_step1000.log      # 合并日志
├── vllm_step1000.log       # vLLM 服务日志
├── eval_step1000.log       # 评估日志
└── ...
```

## 评估指标

评估完成后会显示：

```
Checkpoint  |  F1    |  EM    |  Sub-EM
------------|--------|--------|--------
step_1000   | 75.23  | 68.45  | 72.10
step_2000   | 78.91  | 72.33  | 75.67
```

指标说明：
- **F1**: F1 分数（精确率和召回率的调和平均）
- **EM**: 完全匹配率（Exact Match）
- **Sub-EM**: 子串完全匹配率

## 常见问题

### 1. GPU 显存不足

**现象**: vLLM 启动失败或 OOM

**解决方案**:
```bash
# 编辑 eval_config_7B_4gpu.rc
export VLLM_GPU_MEMORY_UTIL="0.70"  # 降低显存利用率（默认 0.85）
export VLLM_MAX_MODEL_LEN="3072"     # 降低最大长度（默认 4096）
```

### 2. vLLM 端口占用

**现象**: `Address already in use`

**解决方案**:
```bash
# 查找并关闭占用端口的进程
lsof -i :8004
kill -9 <PID>

# 或修改配置使用其他端口
export VLLM_PORT="8005"
```

### 3. 合并速度慢

**现象**: 合并 FSDP checkpoint 耗时较长

**说明**:
- 7B 模型的 4 卡 FSDP checkpoint 总大小约 44GB
- 合并过程需要加载所有分片并重组，预计 5-10 分钟
- 建议使用 `KEEP_MERGED_MODELS="yes"` 保留合并结果

### 4. 评估参数不匹配

**现象**: 评估结果异常或报错

**检查**:
```bash
# 确认配置与训练脚本一致
# eval_config_7B_4gpu.rc:
RECURRENT_CHUNK_SIZE="1536"   # 必须与 7Bnodes.sh line 72 一致
RECURRENT_MAX_NEW="64"         # 必须与 7Bnodes.sh line 74-75 一致
```

## 高级配置

### 并发评估

```bash
# 编辑 eval_config_7B_4gpu.rc
export EVAL_N_PROC="64"  # 增加并发请求数（默认 32）
```

注意：并发数过高可能导致 vLLM OOM

### 修改评估 API 模式

```bash
# 编辑 eval_config_7B_4gpu.rc
export EVAL_API="recurrent"        # 默认：递归式评估
# export EVAL_API="recurrent-boxed"  # 要求 \boxed{} 格式
# export EVAL_API="boxed"            # 单轮 boxed 格式
```

### 添加新的 checkpoint

```bash
# 训练产生新 checkpoint 后，更新配置
# 编辑 eval_config_7B_4gpu.rc:
export CHECKPOINT_STEPS="1000 2000 3000 4000"
```

## 与其他模型的对比

| 模型 | 配置文件 | GPU | 端口 | Chunk Size | 用途 |
|------|----------|-----|------|------------|------|
| 1.5B KV Cache | eval_config.rc | GPU 6 | 8000 | 1024 | KV cache 模式 |
| 1.5B Text | eval_config_text_1k.rc | GPU 4 | 8003 | 3000 | Text 模式 |
| **7B Text 4-GPU** | **eval_config_7B_4gpu.rc** | **GPU 0** | **8004** | **1536** | **四卡训练结果** |

## 技术细节

### FSDP Checkpoint 合并

使用 `scripts/model_merger.py` 进行合并：
```bash
python scripts/model_merger.py \
    --backend fsdp \
    --hf_model_path /mnt/ssd2/models/Qwen2.5-7B-Instruct \
    --local_dir outputs/text_1k_4gpu_optimized/global_step_1000/actor \
    --target_dir merged_models/7B_text_4gpu_step1000
```

合并过程：
1. 读取所有 `model_world_size_4_rank_*.pt` 分片
2. 根据 FSDP sharding 策略重组权重
3. 保存为标准 HuggingFace 格式
4. 复制 tokenizer 文件

### vLLM 推理配置

7B 模型推理优化：
- `max_model_len=4096`: 足够容纳 chunk_size=1536 + 上下文
- `gpu_memory_utilization=0.85`: 7B 模型显存占用较高
- `dtype=bfloat16`: 与训练保持一致
- 自动调整 GPU 利用率以避免 OOM

## 参考文档

- 训练脚本：`7Bnodes.sh`
- 通用评估脚本：`run_evaluation.sh`
- 模型合并脚本：`scripts/model_merger.py`
- 评估工具：`taskutils/memory_eval/ruler_hqa.py`
- 项目文档：`CLAUDE.md`
