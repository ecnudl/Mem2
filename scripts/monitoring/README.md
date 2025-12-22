# MemAgent Training Monitoring Tools

训练监控工具集，用于实时监控和分析 MemAgent 的训练过程。

## 功能特性

### 1. 实时监控 (`monitor_training.py`)
- 实时追踪训练指标（loss, reward, entropy, KL divergence 等）
- 自动生成训练曲线图
- 支持多种数据源：wandb, 本地日志文件
- 可配置的更新频率

### 2. 训练后分析 (`analyze_training.py`)
- 完整的训练报告生成
- 趋势分析和训练有效性评估
- 综合可视化分析（9 个子图）
- 自动判断训练是否有效

## 安装依赖

```bash
pip install matplotlib numpy scipy wandb
```

## 使用方法

### 方式 1: 使用 WandB 实时监控（推荐）

训练脚本已经配置了 wandb 日志记录。启动训练后，在另一个终端运行：

```bash
# 实时监控
python scripts/monitoring/monitor_training.py \
    --mode wandb \
    --project verl-memagent \
    --run lora_4gpu_balanced_20k_n8 \
    --save-dir ./monitoring_plots

# 或者访问 WandB 网页界面
# https://wandb.ai/your-username/verl-memagent
```

### 方式 2: 从日志文件监控

如果没有 wandb 或想离线使用：

```bash
# 启动训练时重定向输出到日志文件
bash run_memory_7B_lora_4gpu_balanced.sh 2>&1 | tee training.log

# 在另一个终端实时监控日志文件
python scripts/monitoring/monitor_training.py \
    --mode file \
    --log-file training.log \
    --save-dir ./monitoring_plots
```

### 方式 3: 自动检测（最简单）

```bash
# 训练完成后，自动从实验目录检测并分析
python scripts/monitoring/monitor_training.py \
    --mode auto \
    --exp-dir /home/admin123/dl/MemAgent/outputs/lora_4gpu_balanced_20k_n8 \
    --no-follow  # 一次性处理，不跟随
```

## 训练后分析

训练完成后，生成完整的分析报告：

```bash
# 从 wandb 分析
python scripts/monitoring/analyze_training.py \
    --source wandb \
    --project verl-memagent \
    --run lora_4gpu_balanced_20k_n8 \
    --output-dir ./analysis_results

# 从日志文件分析
python scripts/monitoring/analyze_training.py \
    --source file \
    --log-file training.log \
    --output-dir ./analysis_results

# 自动检测
python scripts/monitoring/analyze_training.py \
    --source auto \
    --exp-dir /home/admin123/dl/MemAgent/outputs/lora_4gpu_balanced_20k_n8 \
    --output-dir ./analysis_results
```

## 输出说明

### 实时监控输出

- `monitoring_plots/training_curves.png`: 4 个子图的实时训练曲线
  - 左上：Actor Loss 和 Critic Loss
  - 右上：Train 和 Val Reward（带标准差）
  - 左下：KL Divergence 和 Entropy
  - 右下：Validation Accuracy

### 训练后分析输出

- `analysis_results/comprehensive_analysis.png`: 9 个子图的完整分析
  - Actor Loss（原始+平滑）
  - Critic Loss（原始+平滑）
  - Reward Mean（原始+平滑）
  - Reward Distribution（均值±标准差）
  - KL Divergence
  - Policy Entropy
  - Validation Reward
  - Validation Accuracy
  - Loss vs Reward 相关性散点图

- `analysis_results/training_report.json`: 详细的数值分析报告
  - 每个指标的趋势（increasing/decreasing/stable）
  - 起始值、结束值、总体变化
  - 训练有效性评分

## 判断训练有效性的标准

训练有效性通过以下 4 个检查项评估：

1. ✓ **Actor Loss 下降**：损失函数应该逐步降低
2. ✓ **Reward 上升**：平均奖励应该波动式上升
3. ✓ **Critic Loss 稳定/下降**：Critic 应该收敛
4. ✓ **Entropy 未崩溃**：策略熵应保持在合理范围（> 0.01）

如果满足 3-4 项，训练被认为是有效的。
如果仅满足 2 项，训练显示混合结果。
如果少于 2 项，训练可能无效，需要调整超参数。

## 示例：完整训练监控流程

```bash
# 终端 1: 启动训练（已启用 wandb）
bash run_memory_7B_lora_4gpu_balanced.sh

# 终端 2: 实时监控（可选）
python scripts/monitoring/monitor_training.py \
    --mode wandb \
    --project verl-memagent \
    --run lora_4gpu_balanced_20k_n8

# 训练完成后：生成完整分析
python scripts/monitoring/analyze_training.py \
    --source wandb \
    --project verl-memagent \
    --run lora_4gpu_balanced_20k_n8
```

## 参数说明

### monitor_training.py

- `--mode`: 监控模式 (file/wandb/auto)
- `--log-file`: 日志文件路径（file 模式）
- `--project`: WandB 项目名（wandb 模式）
- `--run`: WandB 运行名称（wandb 模式）
- `--exp-dir`: 实验目录（auto 模式）
- `--save-dir`: 图表保存目录
- `--update-interval`: 更新间隔（秒），默认 10
- `--no-follow`: 不跟随日志文件，处理一次后退出

### analyze_training.py

- `--source`: 数据源 (file/wandb/auto)
- `--log-file`: 日志文件路径（file 源）
- `--project`: WandB 项目名（wandb 源）
- `--run`: WandB 运行名称（wandb 源）
- `--exp-dir`: 实验目录（auto 源）
- `--output-dir`: 分析结果保存目录

## 常见问题

### Q: WandB 需要登录吗？
A: 是的，首次使用需要运行 `wandb login` 并输入 API key。可以在 https://wandb.ai/settings 获取。

### Q: 如果没有 wandb 怎么办？
A: 可以使用日志文件模式，只需要将训练输出重定向到文件即可。

### Q: 监控脚本占用资源多吗？
A: 非常少，主要在生成图表时使用 CPU。默认 10 秒更新一次，可以调整 `--update-interval`。

### Q: 如何在远程服务器上使用？
A: 使用 WandB 最方便，可以在本地浏览器查看。或者定期 scp 下载生成的图片。

## 技术细节

- 使用 matplotlib 的 Agg 后端，适合无显示服务器环境
- 支持 Savitzky-Golay 滤波器平滑曲线
- 自动检测可用的指标并绘图
- 线程安全的文件读取（支持日志文件追加写入）

## 示例输出

训练有效的典型输出：

```
==============================================================================
TRAINING ANALYSIS REPORT
==============================================================================

Total training steps: 830

ACTOR_LOSS:
  Trend: decreasing
  Start value: 0.4523
  End value: 0.2145
  Total change: -0.2378

REWARD_MEAN:
  Trend: increasing
  Start value: 0.3421
  End value: 0.7856
  Total change: +0.4435

==============================================================================
TRAINING EFFECTIVENESS ASSESSMENT
==============================================================================
  ✓ Actor loss is decreasing
  ✓ Reward is increasing
  ✓ Critic loss is stable/decreasing
  ✓ No entropy collapse

Overall effectiveness score: 4/4
✓ Training appears to be EFFECTIVE
==============================================================================
```

## 贡献

如有问题或改进建议，欢迎提 issue 或 PR。
