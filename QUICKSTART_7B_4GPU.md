# 7B 四卡训练模型评估 - 快速开始

## 已创建的文件

我为你创建了以下文件来评估你的四卡训练结果：

1. **eval_config_7B_4gpu.rc** - 评估配置文件
2. **eval_7B_4gpu.sh** - 快速启动脚本（推荐使用）
3. **check_eval_env_7B_4gpu.sh** - 环境检查脚本
4. **EVAL_7B_4GPU_README.md** - 详细使用文档

## 使用步骤

### 第一步：检查环境（可选但推荐）

```bash
bash check_eval_env_7B_4gpu.sh
```

这会验证：
- ✓ Conda 环境和 Python 包
- ✓ GPU 配置和显存（使用 GPU 0，训练用的是 1,2,3,4）
- ✓ 基座模型（Qwen2.5-7B-Instruct）
- ✓ Checkpoint 目录（已找到 step_1000 和 step_2000）
- ✓ 参数一致性（chunk_size=1536, max_new=64）

### 第二步：运行评估

**评估所有 checkpoint（推荐第一次运行）：**
```bash
bash eval_7B_4gpu.sh
```

这会自动完成：
1. 合并 FSDP 分片（4个分片 → 完整模型）
2. 启动 vLLM 服务（在 GPU 0 上，端口 8004）
3. 运行评估（100个文档，recurrent 模式）
4. 生成评估报告（F1, EM, Sub-EM 指标）

**评估单个 checkpoint：**
```bash
bash eval_7B_4gpu.sh --step 2000
```

**只合并模型（不评估）：**
```bash
bash eval_7B_4gpu.sh --merge-only
```

**只评估（跳过合并）：**
```bash
bash eval_7B_4gpu.sh --eval-only
```

**自定义文档数量：**
```bash
bash eval_7B_4gpu.sh --length 50   # 快速测试
bash eval_7B_4gpu.sh --length 200  # 更全面的评估
```

## 配置说明

### 关键参数（已自动对齐训练配置）

从你的 `7Bnodes.sh` 训练脚本中提取的参数：

| 参数 | 训练值 | 评估值 | 状态 |
|------|--------|--------|------|
| chunk_size | 1536 | 1536 | ✓ 匹配 |
| max_new_tokens | 64 | 64 | ✓ 匹配 |
| base_model | Qwen2.5-7B-Instruct | Qwen2.5-7B-Instruct | ✓ 匹配 |
| checkpoint_dir | text_1k_4gpu_optimized | text_1k_4gpu_optimized | ✓ 匹配 |

### GPU 分配

- **训练时**: GPU 1, 2, 3, 4（四卡 FSDP）
- **评估时**: GPU 0（单卡推理，避免冲突）

### 端口配置

- vLLM 服务端口: **8004**（避免与其他模型冲突）
- 如果端口被占用，评估脚本会自动关闭旧服务

## 输出结果

### 1. 合并后的模型

```
merged_models/
├── 7B_text_4gpu_step1000/   # 完整的 HuggingFace 格式模型
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── ...
└── 7B_text_4gpu_step2000/
    └── ...
```

### 2. 评估结果

```
taskutils/memory_eval/results/
├── 7B_text_4gpu_step1000/
│   └── eval_100_recurrent.jsonl  # 每个样本的详细结果
├── 7B_text_4gpu_step2000/
│   └── eval_100_recurrent.jsonl
└── summary_7B_text_4gpu_eval_100_recurrent.json  # 汇总报告
```

### 3. 日志文件

```
eval_logs_7B_4gpu/
├── merge_step1000.log       # 合并过程日志
├── vllm_step1000.log        # vLLM 服务日志
├── eval_step1000.log        # 评估运行日志
└── ...
```

### 4. 评估报告示例

评估完成后会显示类似下面的报告：

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

## 预计时间

基于你的配置（7B 模型，4卡 FSDP checkpoint）：

| 步骤 | 预计时间 | 说明 |
|------|----------|------|
| 合并 checkpoint | 5-10 分钟/步 | 需要加载 44GB 分片数据 |
| 启动 vLLM | 1-2 分钟 | 7B 模型加载到 GPU |
| 运行评估（100 docs）| 10-20 分钟 | 取决于 chunk 数量和长度 |
| **总计（2个 checkpoints）** | **约 30-60 分钟** | 首次运行 |

后续运行：
- 如果保留合并模型（`KEEP_MERGED_MODELS="yes"`），跳过合并步骤
- 使用 `--eval-only` 可节省约 10-20 分钟

## 常见问题

### Q1: GPU 显存不足怎么办？

**现象**: vLLM 启动失败，提示 OOM

**解决方案**:
```bash
# 编辑 eval_config_7B_4gpu.rc
export VLLM_GPU_MEMORY_UTIL="0.70"  # 降低显存利用率（默认 0.85）
export VLLM_MAX_MODEL_LEN="3072"     # 降低最大长度（默认 4096）
```

### Q2: 端口被占用怎么办？

**现象**: `Address already in use: 8004`

**解决方案**:
评估脚本会自动关闭旧服务。如果仍有问题：
```bash
# 手动关闭占用端口的进程
lsof -i :8004
kill -9 <PID>
```

### Q3: 合并速度慢？

**说明**:
- 7B 模型的 4 卡 FSDP checkpoint 总大小约 44GB
- 合并过程需要加载所有分片并重组，属于正常现象
- 建议首次运行使用 `KEEP_MERGED_MODELS="yes"`（默认已开启）

### Q4: 如何评估更多 checkpoint？

如果训练产生了新的 checkpoint（如 step_3000）：

```bash
# 编辑 eval_config_7B_4gpu.rc
export CHECKPOINT_STEPS="1000 2000 3000 4000"

# 然后运行评估
bash eval_7B_4gpu.sh
```

## 高级用法

### 修改评估长度

```bash
# 编辑 eval_config_7B_4gpu.rc
export EVAL_LENGTH="200"  # 可选: 50, 100, 200, 400, 800, 1600, 3200, 6400
```

或直接通过命令行：
```bash
bash eval_7B_4gpu.sh --length 200
```

### 修改评估模式

```bash
# 编辑 eval_config_7B_4gpu.rc
export EVAL_API="recurrent"        # 默认：递归式评估
# export EVAL_API="recurrent-boxed"  # 要求 \boxed{} 格式
# export EVAL_API="boxed"            # 单轮 boxed 格式
```

### 强制重新评估

```bash
# 编辑 eval_config_7B_4gpu.rc
export FORCE_MERGE="yes"  # 强制重新合并
export FORCE_EVAL="yes"   # 强制重新评估
```

## 更多信息

- **详细文档**: 查看 `EVAL_7B_4GPU_README.md`
- **配置文件**: 查看 `eval_config_7B_4gpu.rc`
- **训练脚本**: 查看 `7Bnodes.sh`
- **通用评估逻辑**: 查看 `run_evaluation.sh`

## 开始评估

现在你可以直接运行：

```bash
bash eval_7B_4gpu.sh
```

祝评估顺利！如有问题，请查看日志文件：`eval_logs_7B_4gpu/`
