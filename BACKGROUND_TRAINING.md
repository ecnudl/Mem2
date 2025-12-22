# 后台训练使用指南

## ✅ 已完成！训练和监控正在后台运行

训练和监控已经成功启动，即使你关闭 VSCode 或断开 SSH，它们也会继续运行。

## 📊 当前状态

- **训练进程 PID**: 2288119 (✓ 运行中)
- **监控进程 PID**: 2288274 (✓ 运行中)
- **训练日志**: `/home/admin123/dl/MemAgent/training.log`
- **监控日志**: `/home/admin123/dl/MemAgent/monitoring.log`
- **图表输出**: `/home/admin123/dl/MemAgent/monitoring_plots/training_curves.png`

## 🔧 常用命令

### 1. 查看训练状态

```bash
cd /home/admin123/dl/MemAgent
bash scripts/check_training.sh
```

这会显示：
- 进程状态（运行中/已停止）
- 内存和 CPU 使用情况
- 最新的训练进度
- GPU 使用情况

### 2. 实时查看训练日志

```bash
tail -f /home/admin123/dl/MemAgent/training.log
```

按 `Ctrl+C` 退出（不会停止训练）

### 3. 实时查看监控日志

```bash
tail -f /home/admin123/dl/MemAgent/monitoring.log
```

### 4. 查看训练曲线图

在本地（如果使用 SSH）：
```bash
# 从服务器下载到本地
scp username@server:/home/admin123/dl/MemAgent/monitoring_plots/training_curves.png ./
```

在服务器上（如果有图形界面）：
```bash
xdg-open /home/admin123/dl/MemAgent/monitoring_plots/training_curves.png
```

或者使用 VSCode 的远程连接功能直接打开图片。

### 5. 停止训练

```bash
cd /home/admin123/dl/MemAgent
bash scripts/stop_training.sh
```

这会：
- 优雅地停止训练进程
- 停止监控进程
- 清理 Ray 进程
- 保留所有日志和图表

### 6. 重新启动训练

```bash
cd /home/admin123/dl/MemAgent
bash scripts/start_training_with_monitor.sh
```

## 📈 监控说明

### 监控更新频率
- 图表每 **30 秒**自动更新一次
- 监控日志会显示每次更新的时间和最新指标

### 图表内容
`training_curves.png` 包含 4 个子图：
1. **左上**: Actor Loss 和 Critic Loss（双 Y 轴）
2. **右上**: Train 和 Val Reward（带标准差阴影）
3. **左下**: KL Divergence 和 Entropy（双 Y 轴）
4. **右下**: Validation Accuracy

### 判断训练是否有效
观察图表，如果看到：
- ✓ Actor Loss **波动式下降**
- ✓ Reward **波动式上升**
- ✓ Critic Loss **稳定或下降**
- ✓ Entropy **保持合理范围**（不接近 0）

说明训练是有效的！

## 🔍 检查训练进度

### 方式 1: 快速检查

```bash
# 查看最新的几行日志
tail -20 /home/admin123/dl/MemAgent/training.log
```

### 方式 2: 搜索特定步骤

```bash
# 查找包含 "step:100" 的行
grep "step:100" /home/admin123/dl/MemAgent/training.log
```

### 方式 3: 统计训练步数

```bash
# 统计已完成的步数
grep -o "step:[0-9]*" /home/admin123/dl/MemAgent/training.log | tail -1
```

## 📱 远程监控

### 使用 WandB (如果启用)
训练脚本已启用 wandb 日志记录，你可以在任何地方通过浏览器访问：

```
https://wandb.ai/your-username/verl-memagent
```

### 使用 SSH 隧道查看图表
如果想在本地浏览器实时查看图表，可以设置一个简单的 HTTP 服务器：

```bash
# 在服务器上启动简单的 HTTP 服务
cd /home/admin123/dl/MemAgent/monitoring_plots
python3 -m http.server 8888 &

# 在本地电脑创建 SSH 隧道
ssh -L 8888:localhost:8888 username@server

# 然后在本地浏览器访问
http://localhost:8888/training_curves.png
```

## ⚠️ 常见问题

### Q: 如何确认训练还在运行？
```bash
# 检查进程
ps aux | grep "verl.trainer.main_ppo"

# 或者使用状态脚本
bash scripts/check_training.sh
```

### Q: 训练日志太大了怎么办？
训练日志会持续增长。如果担心磁盘空间，可以：
```bash
# 查看日志大小
du -h /home/admin123/dl/MemAgent/training.log

# 截断日志（保留最近 10000 行）
tail -10000 /home/admin123/dl/MemAgent/training.log > temp.log
mv temp.log /home/admin123/dl/MemAgent/training.log
```

### Q: 监控图表没有更新？
检查监控进程是否运行：
```bash
ps aux | grep "monitor_training.py"

# 查看监控日志
tail -20 /home/admin123/dl/MemAgent/monitoring.log
```

如果监控停止了，可以单独重启：
```bash
cd /home/admin123/dl/MemAgent/scripts/monitoring
nohup python3 monitor_training.py \
    --mode file \
    --log-file ../../training.log \
    --save-dir ../../monitoring_plots \
    > ../../monitoring.log 2>&1 &
```

### Q: 磁盘空间警告怎么办？
训练过程中可能出现磁盘空间警告（95% 满），这是正常的。只要还有 200GB+ 可用空间，训练可以继续。

如果需要清理空间：
```bash
# 清理旧的 Ray 临时文件
bash scripts/cleanup_ray_tmp.sh
```

## 📝 训练完成后

训练完成后，运行完整分析：

```bash
cd /home/admin123/dl/MemAgent/scripts/monitoring
bash quick_analyze.sh
```

这会生成：
- `analysis_results/comprehensive_analysis.png` - 9 个详细分析图
- `analysis_results/training_report.json` - 数值分析报告

报告会自动评估训练有效性，给出评分（满分 4 分）。

## 🎉 完成！

现在你可以：
1. ✅ 安心关闭 VSCode - 训练继续运行
2. ✅ 断开 SSH - 训练继续运行
3. ✅ 随时重连查看进度
4. ✅ 随时下载最新的训练曲线图

祝训练顺利！🚀

---

**创建时间**: 2025-12-12 03:29
**训练配置**: Balanced 4GPU LoRA (20k tokens, n=8)
**预计训练时间**: 根据你的数据集大小和 GPU 性能而定
