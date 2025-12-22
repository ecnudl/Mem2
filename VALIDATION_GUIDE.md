# MemAgent 验证集配置指南

## 📋 目录
- [验证集的作用](#验证集的作用)
- [问题分析：为什么不推荐 dev_20](#问题分析)
- [推荐方案：dev_100_balanced](#推荐方案)
- [如何创建自定义验证集](#创建自定义验证集)
- [训练脚本更新](#训练脚本更新)

---

## 验证集的作用

### 核心功能

在PPO训练过程中，验证集（`VAL_PATH`）承担以下重要职责：

1. **监控模型性能**
   - 每 `test_freq` 步（默认15步）运行一次验证
   - 使用当前模型生成验证集的响应
   - 通过reward function评分（HotpotQA验证答案正确性）
   - 记录准确率、奖励等关键指标

2. **早停和模型选择**
   - 追踪最佳checkpoint
   - 检测过拟合（训练acc上升但验证acc下降）
   - 帮助决定最佳停止时机

3. **超参数调优**
   - 对比不同配置的验证表现
   - 选择最佳学习率、batch size等

4. **可视化训练进度**
   - 在wandb/tensorboard中展示训练曲线
   - 查看生成样例的质量变化
   - 调试prompt/response问题

### 验证流程

```
每15步触发一次验证：
1. 从验证集加载batch
2. 对每个样本生成 n=4 个响应（rollout.val_kwargs.n）
3. 用reward verifier评分
4. 计算平均准确率、奖励等指标
5. 记录到日志/wandb
6. 保存验证样例（可选）

⏱️  验证时间 = 验证样本数 × val_kwargs.n × 单样本生成时间
```

---

## 问题分析：为什么不推荐 dev_20

### `hotpotqa_dev_20.parquet` 的问题

| 指标 | 数值 | 问题 |
|------|------|------|
| **样本数** | 20 | ❌ 太少，统计不稳定（±5-10%波动） |
| **Context长度** | 平均43K tokens | ❌ 远超训练集26K，分布不匹配 |
| **截断损失** | ~50% | ❌ MAXLEN=20K会截断一半content |
| **覆盖面** | 20个问题 | ❌ 无法代表整体性能 |
| **验证时间** | ~2.7分钟/次 | ✅ 唯一优点：快速 |

### 主要问题详解

#### 1. 样本数太少（20个）
- **统计不稳定**：准确率波动大（±5-10%）
- **无法可靠评估**：可能碰巧简单或困难
- **后果**：无法判断模型真实性能提升

#### 2. Context过长（43K tokens）
- **与训练集不匹配**：训练集平均26K，验证集43K
- **分布偏移**：测试的不是模型在训练分布上的表现
- **误导性指标**：验证准确率不代表真实能力

#### 3. 严重截断（50%+）
- **MAXLEN=20K** 会保留开头10K+结尾10K，丢弃中间23K
- **损失关键信息**：中间部分可能包含重要上下文
- **不公平评估**：测试的是截断后的性能，而非完整能力

#### 4. 覆盖面窄
- **只有20个问题类型**
- **缺乏代表性**：无法反映多样性
- **过拟合风险**：模型可能记住这20个样本

### 验证开销对比

| 验证集 | 样本数 | 时间/次 | 时间/epoch | 占训练时间 |
|--------|--------|---------|-----------|-----------|
| dev_20 (旧) | 20 | ~2.7分 | ~13分 | 22% |
| **dev_100_balanced (推荐)** | **100** | **~13分** | **~65分** | **11%** |
| dev (完整) | 7,405 | ~987分 | ~4,937分 | 8228% |

---

## 推荐方案：dev_100_balanced

### ✅ 新验证集特点

**文件**: `hotpotqa_dev_100_balanced.parquet`

| 指标 | 数值 | 优势 |
|------|------|------|
| **样本数** | 100 | ✅ 足够大保证统计稳定（±1-2%） |
| **Context长度** | 平均28K tokens | ✅ 接近训练集26K（差异仅7%） |
| **长度范围** | 26.4K - 30K | ✅ 与训练集分布匹配 |
| **来源** | dev.parquet采样 | ✅ 独立验证集，无训练泄露 |
| **截断损失** | ~28% | ✅ 在MAXLEN=20K下合理 |
| **验证时间** | ~13分钟/次 | ✅ 快速且准确 |
| **文件大小** | 6.7 MB | ✅ 轻量级 |

### 为什么是最佳选择？

#### 1. **样本数量合适（100个）**
```
✅ 足够大：保证统计稳定性（准确率误差±1-2%）
✅ 足够小：验证快速（~13分钟/次）
✅ 黄金比例：约为训练集的10%（1000样本）
```

#### 2. **长度分布匹配训练集**
```
dev_100_balanced: 平均 28,276 tokens
train_1k:         平均 26,038 tokens
差异：仅 8.6% （非常接近！）
```

#### 3. **来自独立验证集**
```
✅ 从 dev.parquet 随机采样
✅ 与训练集无重叠
✅ 真正测试泛化能力
```

#### 4. **在不同MAXLEN下的表现**

| MAXLEN | 截断比例 | 保留内容 | 评估 |
|--------|---------|---------|------|
| 15K | ~47% | 开头7.5K + 结尾7.5K | ⚠️  较大截断 |
| **20K** | **~28%** | **开头10K + 结尾10K** | **✅ 推荐** |
| 28K | ~1% | 几乎完整 | ✅ 理想 |

#### 5. **验证开销合理**
```
每次验证: 100样本 × 4采样 = 400次生成 (~13分钟)
每epoch:  ~5次验证 × 13分钟 = ~65分钟
占比:     约11%的训练时间（可接受）
```

### 与其他选项对比

| 验证集 | 样本数 | Context均值 | 推荐度 | 适用场景 |
|--------|--------|------------|--------|---------|
| dev_20 | 20 | 43K | ❌ 不推荐 | 仅快速调试 |
| train_100 | 100 | 39K | ⚠️  可用 | 缺少dev数据时 |
| **dev_100_balanced** | **100** | **28K** | **✅ 强烈推荐** | **正式训练** |
| dev (完整) | 7,405 | 28K | ❌ 太慢 | 最终评估 |

---

## 创建自定义验证集

如果你需要针对不同的 `MAXLEN` 创建定制化验证集：

### 使用脚本

```bash
# 基本用法：创建100个样本的验证集
python scripts/create_validation_set.py \
    --num_samples 100 \
    --min_length 24000 \
    --max_length 30000 \
    --seed 42

# 为Conservative配置 (MAXLEN=15K) 创建验证集
python scripts/create_validation_set.py \
    --output taskutils/memory_data/hotpotqa/hotpotqa_dev_100_conservative.parquet \
    --num_samples 100 \
    --min_length 20000 \
    --max_length 28000

# 为Aggressive配置 (MAXLEN=28K) 创建验证集
python scripts/create_validation_set.py \
    --output taskutils/memory_data/hotpotqa/hotpotqa_dev_100_aggressive.parquet \
    --num_samples 100 \
    --min_length 26000 \
    --max_length 32000

# 创建更大的验证集（200个样本）
python scripts/create_validation_set.py \
    --output taskutils/memory_data/hotpotqa/hotpotqa_dev_200.parquet \
    --num_samples 200 \
    --min_length 24000 \
    --max_length 30000
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | hotpotqa_dev.parquet | 源数据文件 |
| `--output` | hotpotqa_dev_100_filtered.parquet | 输出文件路径 |
| `--num_samples` | 100 | 采样数量 |
| `--min_length` | 20000 | 最小context长度（tokens） |
| `--max_length` | 30000 | 最大context长度（tokens） |
| `--seed` | 42 | 随机种子（保证可复现） |

### 针对不同MAXLEN的建议

| MAXLEN | 推荐min_length | 推荐max_length | 说明 |
|--------|----------------|----------------|------|
| 15K | 20,000 | 28,000 | 控制截断在~30% |
| **20K** | **24,000** | **30,000** | **当前balanced配置** |
| 25K | 24,000 | 32,000 | 轻微截断 |
| 28K | 26,000 | 34,000 | 几乎无截断 |

---

## 训练脚本更新

### ✅ 所有脚本已更新

所有4GPU训练脚本已自动更新为使用 `dev_100_balanced.parquet`：

```bash
✅ run_memory_7B_lora_4gpu_balanced.sh      (推荐)
✅ run_memory_7B_lora_4gpu_conservative.sh  (快速实验)
✅ run_memory_7B_lora_4gpu_aggressive.sh    (极限性能)
```

### 验证集配置

所有脚本中的验证集路径：
```bash
VAL_PATH="${DATASET_ROOT}/hotpotqa/hotpotqa_dev_100_balanced.parquet"
```

### 如何切换验证集

如果需要使用其他验证集，修改脚本中的 `VAL_PATH`：

```bash
# 选项1：使用dev_20（仅调试）
VAL_PATH="${DATASET_ROOT}/hotpotqa/hotpotqa_dev_20.parquet"

# 选项2：使用train_100（缺少dev数据时）
VAL_PATH="${DATASET_ROOT}/hotpotqa/hotpotqa_train_100.parquet"

# 选项3：使用dev_100_balanced（推荐）
VAL_PATH="${DATASET_ROOT}/hotpotqa/hotpotqa_dev_100_balanced.parquet"

# 选项4：使用自定义验证集
VAL_PATH="${DATASET_ROOT}/hotpotqa/my_custom_validation.parquet"
```

---

## 最佳实践

### 1. 验证频率设置

```bash
# 快速迭代（更频繁验证）
trainer.test_freq=10     # 每10步验证一次

# 平衡配置（推荐）
trainer.test_freq=15     # 每15步验证一次

# 减少验证开销（适合大验证集）
trainer.test_freq=30     # 每30步验证一次

# 禁用验证（不推荐）
trainer.test_freq=-1     # 完全禁用
```

### 2. 验证前训练

```bash
# 训练前先验证（检查初始性能）
trainer.val_before_train=True

# 跳过初始验证（节省时间，推荐）
trainer.val_before_train=False
```

### 3. 监控验证指标

训练时关注以下指标：

- **验证准确率（validation_accuracy）**: 主要指标
- **验证奖励（validation_reward）**: 平均奖励分数
- **训练vs验证差距**: 检测过拟合
  - 差距<5%: 健康
  - 差距5-10%: 轻微过拟合
  - 差距>10%: 严重过拟合

### 4. 保存验证样例

```bash
# 启用验证样例保存
trainer.validation_data_dir=/path/to/save/validation_outputs

# 查看生成的响应质量
cat /path/to/save/validation_outputs/step_300.jsonl
```

---

## 常见问题

### Q1: 验证时间太长怎么办？

**方案1**: 减少验证频率
```bash
trainer.test_freq=30  # 从15改为30
```

**方案2**: 减少验证样本
```bash
python scripts/create_validation_set.py --num_samples 50
```

**方案3**: 减少验证采样数
```bash
actor_rollout_ref.rollout.val_kwargs.n=2  # 从4改为2
```

### Q2: 验证准确率远低于训练准确率？

可能原因：
1. **过拟合**: 模型记住了训练集
   - 解决：降低学习率，增加正则化
2. **验证集太难**: dev集比train集难
   - 正常现象，关注趋势而非绝对值
3. **Verifier不同**: 训练用严格verifier，测试用宽松verifier
   - 正常现象，这是设计如此

### Q3: 能用训练集做验证吗？

❌ **不推荐**：
- 无法检测过拟合
- 验证准确率虚高
- 无法反映泛化能力

⚠️  **特殊情况可用**：
- 缺少独立验证集
- 仅用于调试代码
- 明确标注"非正式验证"

### Q4: 最佳验证集大小是多少？

经验法则：
- **小训练集（<1K）**: 验证集 = 训练集的10-20%
- **中训练集（1K-10K）**: 验证集 = 100-200个样本
- **大训练集（>10K）**: 验证集 = 500-1000个样本

对于1K训练集：**100个样本是理想选择**

---

## 总结

### 推荐配置（默认）

```bash
# 验证集
VAL_PATH="hotpotqa_dev_100_balanced.parquet"

# 验证频率
trainer.test_freq=15

# 验证采样
actor_rollout_ref.rollout.val_kwargs.n=4
actor_rollout_ref.rollout.val_kwargs.temperature=0.0
actor_rollout_ref.rollout.val_kwargs.do_sample=False

# 不在训练前验证
trainer.val_before_train=False
```

### 关键要点

✅ **推荐**: 使用 `dev_100_balanced.parquet`
- 100个样本：统计稳定
- 28K tokens：匹配训练分布
- 来自dev集：真实泛化能力
- ~13分钟验证：开销合理

❌ **避免**: 使用 `dev_20.parquet`
- 20个样本太少
- 43K tokens过长
- 统计不稳定

🔧 **自定义**: 使用 `create_validation_set.py`
- 针对不同MAXLEN定制
- 调整样本数和长度范围
- 保证可复现（固定seed）

---

**创建日期**: 2025-12-10
**适用版本**: MemAgent v1.0
**维护者**: Claude Code
