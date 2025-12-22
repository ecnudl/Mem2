# 训练监控快速开始指南

## 🎯 目标

本指南帮助你快速开始监控 MemAgent 的训练过程，实时查看 loss 和 reward 曲线，验证训练有效性。

## ✅ 前置准备

```bash
# 安装依赖（如果还没有安装）
pip install matplotlib numpy scipy

# 可选：安装 wandb 用于更好的体验
pip install wandb
wandb login  # 首次使用需要登录
```

## 🚀 三步开始

### 步骤 1: 启动训练（已配置好日志记录）

```bash
bash run_memory_7B_lora_4gpu_balanced.sh
```

训练脚本已自动配置了 wandb 和 console 日志。

### 步骤 2: 实时监控（在另一个终端）

**方式 A - 使用 WandB（推荐）：**

```bash
cd /home/admin123/dl/MemAgent/scripts/monitoring
./quick_monitor.sh
```

然后选择选项 1（WandB），或者访问：https://wandb.ai/your-username/verl-memagent

**方式 B - 使用日志文件：**

```bash
# 终端 1：启动训练并记录日志
bash run_memory_7B_lora_4gpu_balanced.sh 2>&1 | tee training.log

# 终端 2：监控日志
cd /home/admin123/dl/MemAgent/scripts/monitoring
python3 monitor_training.py --mode file --log-file ../../training.log
```

监控脚本会每 10 秒更新一次图表，保存在 `monitoring_plots/training_curves.png`。

### 步骤 3: 训练完成后分析

```bash
cd /home/admin123/dl/MemAgent/scripts/monitoring
./quick_analyze.sh
```

查看生成的分析报告：
- `analysis_results/comprehensive_analysis.png` - 9 个详细图表
- `analysis_results/training_report.json` - 数值分析结果

## 📊 如何判断训练是否有效

训练有效的标志：

1. **✓ Actor Loss 波动下降**
   - 初始值高（如 0.5）
   - 逐步降低到较低值（如 0.2-0.3）
   - 允许有波动，但整体趋势向下

2. **✓ Reward 波动上升**
   - 初始值低（如 0.3）
   - 逐步上升到较高值（如 0.7-0.8）
   - 允许有波动，但整体趋势向上

3. **✓ Critic Loss 稳定或下降**
   - 不应该持续上升
   - 可以波动，但不应发散

4. **✓ Entropy 不崩溃**
   - 保持在合理范围（> 0.01）
   - 不应该降至接近 0

**分析脚本会自动检查这 4 个标准，给出有效性评分。**

满足 3-4 个标准 → ✓ 训练有效
满足 2 个标准 → ⚠ 混合结果
满足 0-1 个标准 → ✗ 训练无效

## 🔧 常见使用场景

### 场景 1: 本地训练，实时监控

```bash
# 终端 1
bash run_memory_7B_lora_4gpu_balanced.sh 2>&1 | tee training.log

# 终端 2
cd scripts/monitoring
python3 monitor_training.py --mode file --log-file ../../training.log
```

### 场景 2: 远程服务器训练，使用 WandB

```bash
# 服务器上启动训练（已配置 wandb）
bash run_memory_7B_lora_4gpu_balanced.sh

# 本地浏览器访问
https://wandb.ai/your-username/verl-memagent
```

### 场景 3: 训练完成后分析

```bash
cd scripts/monitoring
python3 analyze_training.py \
    --source auto \
    --exp-dir ../../outputs/lora_4gpu_balanced_20k_n8
```

## 📈 示例输出

### 实时监控输出

```
[2025-12-12 02:36:55] Plot updated: monitoring_plots/training_curves.png
  Steps: 450, Last step: 450
  Latest train reward: 0.6523
  Latest val reward: 0.7234
```

### 训练后分析输出

```
==============================================================================
TRAINING ANALYSIS REPORT
==============================================================================

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

TRAINING EFFECTIVENESS ASSESSMENT:
  ✓ Actor loss is decreasing
  ✓ Reward is increasing
  ✓ Critic loss is stable/decreasing
  ✓ No entropy collapse

Overall effectiveness score: 4/4
✓ Training appears to be EFFECTIVE
==============================================================================
```

## 🛠️ 高级选项

### 自定义更新频率

```bash
python3 monitor_training.py \
    --mode file \
    --log-file training.log \
    --update-interval 30  # 30秒更新一次（默认10秒）
```

### 自定义保存目录

```bash
python3 monitor_training.py \
    --mode wandb \
    --project verl-memagent \
    --run lora_4gpu_balanced_20k_n8 \
    --save-dir /path/to/custom/plots
```

### 一次性处理（不跟随日志）

```bash
python3 monitor_training.py \
    --mode file \
    --log-file training.log \
    --no-follow  # 处理现有内容后退出
```

## ❓ 常见问题

### Q: 图表不更新怎么办？

A: 检查几点：
1. 训练是否在正常运行？
2. 日志文件是否在增长？（`tail -f training.log`）
3. 监控脚本是否有报错？

### Q: wandb 连接不上？

A: 使用日志文件模式作为备选方案，完全离线可用。

### Q: 如何在 tmux/screen 中使用？

```bash
# Session 1: 训练
tmux new -s training
bash run_memory_7B_lora_4gpu_balanced.sh 2>&1 | tee training.log

# Session 2: 监控（Ctrl+B, D 分离后创建新会话）
tmux new -s monitor
cd scripts/monitoring
python3 monitor_training.py --mode file --log-file ../../training.log
```

### Q: 图表保存在哪里？

- 实时监控: `monitoring_plots/training_curves.png`
- 训练后分析: `analysis_results/comprehensive_analysis.png`

### Q: 如何查看远程服务器上的图表？

**方式 1 - SCP 下载：**
```bash
scp username@server:/path/to/monitoring_plots/training_curves.png .
```

**方式 2 - 使用 WandB：**
网页端实时查看，无需下载。

**方式 3 - 使用 VSCode Remote：**
如果使用 VSCode 连接服务器，可以直接在侧边栏预览图片。

## 📚 更多帮助

详细文档：`scripts/monitoring/README.md`

脚本说明：
- `monitor_training.py` - 实时监控脚本
- `analyze_training.py` - 训练后分析脚本
- `quick_monitor.sh` - 快速启动监控
- `quick_analyze.sh` - 快速启动分析
- `test_monitoring.py` - 测试工具功能

测试工具：
```bash
cd scripts/monitoring
python3 test_monitoring.py  # 验证所有功能正常
```

## 🎉 开始使用

现在你可以开始训练并监控了！

```bash
# 1. 启动训练
bash run_memory_7B_lora_4gpu_balanced.sh

# 2. 在另一个终端监控
cd scripts/monitoring && ./quick_monitor.sh

# 3. 训练完成后分析
cd scripts/monitoring && ./quick_analyze.sh
```

祝训练顺利！🚀
