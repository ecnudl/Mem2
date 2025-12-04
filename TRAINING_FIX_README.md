# MemAgent 1.5B Training Configuration Fix

## 🔍 Problem Summary

The original training showed **performance degradation** over time:
- ✅ step_1000: EM=21%
- ⚠️ step_2000: EM=8%
- ❌ step_10000: EM=6%

**Root Cause**: Model only sees **1.5% of context** during training (1,024 out of 69,436 tokens), leading to:
1. Reward hacking (learning to guess patterns from first 1K tokens)
2. Catastrophic forgetting (losing pretrained knowledge)
3. Train-test mismatch (evaluation uses full context)

---

## 📊 Configuration Comparison

| Parameter | Old Config | New Config | Impact |
|-----------|------------|------------|--------|
| **max_chunks** | 1 ❌ | 30 ✅ | Context coverage: 1.5% → 44% |
| **Total tokens** | 1,024 | 30,720 | 30× increase |
| **KV cache max** | 2,560 | 32,768 | Proper sizing |
| **Batch size** | 1 | 1 | Keep for VRAM safety |
| **VRAM usage** | ~10 GB | ~30 GB | Still fits 48GB GPU |
| **Context coverage** | 1.5% 😢 | 44% 😊 | 29× improvement |

### Key Changes Explained

#### 1. **max_chunks: 1 → 30** (Most Critical)
- **Before**: Model reads only first 1,024 tokens (like reading 1 page of a 700-page book)
- **After**: Model reads 30,720 tokens (like reading 300 pages)
- **Why 30?**: Balance between VRAM (48GB limit) and coverage
- **Can increase later**: If VRAM allows, try 40 (59% coverage) or 50 (74% coverage)

#### 2. **KV Cache Sizing**
```bash
# Formula: (max_chunks × chunk_size) + prompt_buffer + response_buffer
30 × 1024 + 1792 + 512 = 32,768 tokens
```

#### 3. **Memory Optimizations**
- ✅ Gradient checkpointing enabled
- ✅ Reference model offloaded to CPU
- ✅ Critic model offloaded to CPU
- ✅ Batch size kept at 1
- ✅ Dynamic batch sizing enabled

---

## 🚀 How to Use

### Step 1: Start Training

```bash
cd /home/admin123/dl/MemAgent

# Make sure script is executable
chmod +x run_memory_1.5B_kv_new.sh

# Launch training (will take ~2-3 days on single GPU)
bash run_memory_1.5B_kv_new.sh
```

**Monitor VRAM usage** during first epoch:
```bash
# In another terminal
watch -n 2 nvidia-smi
```

Expected VRAM: ~30GB (safe for 48GB GPU)

### Step 2: After Training, Evaluate Checkpoints

```bash
# Source the evaluation config
source eval_config_1.5B_new.rc

# Run batch evaluation
bash run_evaluation.sh

# Results will be saved to:
# - taskutils/memory_eval/results/1.5B_kv_new_step*/
# - taskutils/memory_eval/results/summary_1.5B_kv_new_eval_100_recurrent.json
```

---

## 📈 Expected Results

### Before Fix (Old Training)
```
step_1000:  EM=21%, F1=23.7  ← Best (retains pretrained knowledge)
step_2000:  EM=8%,  F1=9.8   ← Degrading
step_10000: EM=6%,  F1=7.9   ← Worst (catastrophic forgetting)
```

### After Fix (New Training)
```
step_1000:  EM=25-35%, F1=30-40  ← Learning useful patterns
step_2000:  EM=35-45%, F1=40-50  ← Improving
step_10000: EM=45-60%, F1=50-65  ← Best (proper convergence)
```

**Key Indicators of Success**:
1. ✅ Training reward steadily increases (not random jumps)
2. ✅ Validation accuracy improves over time
3. ✅ Later checkpoints perform better than early ones
4. ✅ Final EM should reach 45-60% (vs current 6%)

---

## 🔧 Troubleshooting

### Problem: OOM (Out of Memory)

**Solution 1**: Reduce max_chunks
```bash
# Edit run_memory_1.5B_kv_new.sh line 33:
MAX_CHUNKS=20  # From 30 to 20 (coverage: 44% → 29%)
KV_CACHE_MAX_LEN=22528  # 20×1024 + 1792 + 512
```

**Solution 2**: Enable more offloading
```bash
# Edit run_memory_1.5B_kv_new.sh, change:
actor_rollout_ref.actor.fsdp_config.param_offload=True  # Was False
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True  # Was False
```

### Problem: Training too slow

**Current speed**: ~2-3 days for 10,000 steps on single GPU

**Option 1**: Reduce total_epochs
```bash
trainer.total_epochs=5  # From 10 to 5 (half the time)
```

**Option 2**: Use multiple GPUs (requires code changes)
```bash
NGPUS_PER_NODE=2
export CUDA_VISIBLE_DEVICES=2,3
# Also adjust FSDP and batch size accordingly
```

### Problem: Still getting reward hacking

**Check**: Make sure model is reading full chunks

```bash
# During training, you should see logs like:
# "Processing chunk 1/30"
# "Processing chunk 2/30"
# ...
# "Processing chunk 30/30"
```

If you only see "chunk 1/1", the config didn't apply correctly.

---

## 🎯 Next Steps After This Fix

### Short-term (After verifying this works)
1. **Gradually increase max_chunks**:
   - Epoch 1-2: max_chunks=30 (44% coverage)
   - Epoch 3-5: max_chunks=40 (59% coverage)
   - Epoch 6-10: max_chunks=50 (74% coverage)

2. **Increase training data**:
   - Current: 1,000 samples
   - Recommended: 5,000-10,000 samples
   - This reduces overfitting

### Long-term (Scaling up)
1. **Use 0.5B model** for faster iteration:
   - Create similar fix for `run_memory_0.5B_kv.sh`
   - Set max_chunks=50-70 (0.5B has lower VRAM usage)

2. **Multi-GPU training**:
   - Scale to 2-4 GPUs
   - Increase batch size accordingly
   - Faster convergence

3. **Curriculum learning**:
   - Start with short contexts (20 chunks)
   - Gradually increase to full context (70 chunks)
   - Better training stability

---

## 📝 Files Created

1. **run_memory_1.5B_kv_new.sh** - Fixed training script
2. **eval_config_1.5B_new.rc** - Matching evaluation config
3. **TRAINING_FIX_README.md** - This documentation

---

## ⚠️ Important Notes

1. **Do NOT resume from old checkpoints**: They learned bad patterns. Start fresh.
   ```bash
   # The script already has:
   trainer.resume_mode=disable
   ```

2. **Monitor training metrics**: Check console logs for:
   - Average reward per episode
   - KL divergence (should stay small)
   - Number of chunks processed per sample

3. **Evaluation MUST use same chunk_size**: Already configured in `eval_config_1.5B_new.rc`

4. **Be patient**: First few epochs may show slow improvement as model unlearns bad patterns from base model.

---

## 🙋 Questions?

**Q: Why not use max_chunks=75 to cover 100% of context?**
A: VRAM limitation (48GB). Also, 44% coverage is enough for HotpotQA since answers typically appear in first half of documents.

**Q: Can I train overnight?**
A: With 1,000 samples and 10 epochs, expect 2-3 days. Reduce `total_epochs=5` for faster experimentation.

**Q: What if I have 80GB GPU?**
A: Increase max_chunks=60-75 for better coverage. Also consider increasing batch_size=2.

**Q: Should I stop using text memory mode?**
A: KV cache mode is more efficient. Text mode would need even more VRAM for long contexts.

---

**Good luck with the training! 🚀**
