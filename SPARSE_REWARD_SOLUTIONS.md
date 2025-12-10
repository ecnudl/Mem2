# Sparse Reward Problem - Complete Solution Guide

## Problem Summary

**Current Status:**
- Training reward: ~0% samples get non-zero reward
- Validation accuracy: 5-15% (random guessing level)
- Model behavior: Answers "\boxed{No information available}" instead of "yes"/"no"

**Root Cause:**
1. Model hasn't learned to use memory mechanism to accumulate information
2. Binary reward (0 or 1) provides too sparse signal for PPO
3. Base model tends to give conservative "I don't know" answers
4. Learning rate may be too high (5e-5 vs 1e-6 for full fine-tuning)

---

## Solution Strategies (Ordered by Recommendation)

### ✅ Solution 1: Shaped Reward (RECOMMENDED - Easiest)

**What it does:**
Provides intermediate reward signals instead of binary 0/1:
- **Format reward (+0.2)**: Uses `\boxed{}` format
- **Attempt reward (+0.3)**: Gives an answer (not "unknown"/"cannot determine")
- **Correct reward (+0.5)**: Correct answer

Total possible reward: 0.0 to 1.0 (continuous instead of binary)

**How to use:**
```bash
# Already created for you!
bash run_memory_7B_lora_shaped_reward.sh
```

**Key changes:**
- Learning rate: 5e-5 → 1e-5 (10x reduction for stability)
- LoRA rank: 64 → 32 (better stability)
- LoRA dropout: 0.0 → 0.05 (regularization)
- Shaped reward enabled via `USE_SHAPED_REWARD=true`

**Expected improvement:**
- First 100 steps: format_reward should increase from 0% → 50%+
- By 500 steps: attempt_reward should show progress
- By 1000 steps: correct_reward should start appearing

**Monitor progress:**
```bash
# Watch training logs
tail -f outputs/lora_1k_2gpu_r32_shaped_lr1e5/latest.log

# Look for lines like:
# [format_reward] 0.2
# [attempt_reward] 0.3
# [correct_reward] 0.5
# [score] 1.0
```

---

### Solution 2: Curriculum Learning

**Idea:** Start with easier examples, gradually increase difficulty

**Implementation:**
```bash
# Create curriculum datasets
python3 << 'EOF'
import pandas as pd

# Load training data
df = pd.read_parquet('/home/admin123/dl/MemAgent/taskutils/memory_data/hotpotqa/hotpotqa_train.parquet')

# Stage 1: Short contexts (easiest)
df_easy = df[df['context'].str.len() < 1000].head(500)
df_easy.to_parquet('hotpotqa_train_easy.parquet')

# Stage 2: Medium contexts
df_medium = df[(df['context'].str.len() >= 1000) & (df['context'].str.len() < 2000)].head(500)
df_medium.to_parquet('hotpotqa_train_medium.parquet')

# Stage 3: Hard contexts
df_hard = df[df['context'].str.len() >= 2000].head(500)
df_hard.to_parquet('hotpotqa_train_hard.parquet')
EOF

# Train in stages
TRAIN_PATH=hotpotqa_train_easy.parquet bash run_memory_7B_lora_shaped_reward.sh
# After converging, switch to medium, then hard
```

---

### Solution 3: Behavior Cloning Pre-training

**Idea:** First do supervised fine-tuning on correct examples, then do PPO

**Step 1: Create SFT dataset**
```python
# Create supervised learning data with correct demonstrations
import pandas as pd

df = pd.read_parquet('hotpotqa_train_1k.parquet')

# Add correct answer demonstrations
sft_data = []
for _, row in df.iterrows():
    prompt = row['prompt']  # The multi-turn memory prompt
    answer = row['answer']   # Ground truth
    # Format: prompt → "\boxed{answer}"
    sft_data.append({
        'prompt': prompt,
        'completion': f"\\boxed{{{answer}}}"
    })

pd.DataFrame(sft_data).to_parquet('hotpotqa_sft_1k.parquet')
```

**Step 2: Run SFT first**
```bash
# Use Hugging Face Trainer or similar
# Train for 1-2 epochs on correct demonstrations
# This teaches the model the task structure
```

**Step 3: Then run PPO**
```bash
# Now PPO starts from a better initialization
bash run_memory_7B_lora_shaped_reward.sh
```

---

### Solution 4: Increase Training Data

**Current:** 1000 samples
**Recommended:** 3200+ samples

```bash
# Modify training script
TRAIN_PATH="${DATASET_ROOT}/hotpotqa/hotpotqa_train_3200.parquet"

# Or create it:
python3 /home/admin123/dl/MemAgent/scripts/create_train_1k.py \
    --input "${DATASET_ROOT}/hotpotqa/hotpotqa_train.parquet" \
    --output "${DATASET_ROOT}/hotpotqa/hotpotqa_train_3200.parquet" \
    --num_samples 3200
```

More data → more diverse examples → better learning signal

---

### Solution 5: Adjust Memory Configuration

**Hypothesis:** Model can't access needed information due to memory limits

**Current config:**
```python
chunk_size=1536          # Characters per chunk
max_chunks=16            # Maximum memory turns
max_memorization_length=64   # Tokens to generate per memory turn
```

**Try increasing context window:**
```bash
# In your training script, change:
recurrent.memory.config.chunk_size=2048 \    # Was 1536
recurrent.memory.config.max_chunks=20 \      # Was 16
```

**Or try smaller chunks (more turns):**
```bash
recurrent.memory.config.chunk_size=1024 \    # Smaller chunks
recurrent.memory.config.max_chunks=24 \      # More turns
```

---

### Solution 6: Add Exploration Bonus

**Idea:** Reward the model for trying different answers (not always "no information")

Create `hotpotqa_exploration.py`:
```python
def compute_score(solution_str, ground_truth) -> dict:
    from .hotpotqa_shaped import compute_score as shaped_score

    rewards = shaped_score(solution_str, ground_truth)

    # Exploration bonus: penalize repetitive "no information" answers
    if 'no information' in solution_str.lower():
        rewards['exploration_penalty'] = -0.1
    else:
        rewards['exploration_penalty'] = 0.0

    rewards['score'] += rewards['exploration_penalty']
    return rewards
```

---

## Quick Start Guide

### Option A: Try Shaped Reward (5 minutes)

```bash
# 1. Start training with shaped reward
cd /home/admin123/dl/MemAgent
bash run_memory_7B_lora_shaped_reward.sh

# 2. Monitor progress (in another terminal)
watch -n 10 "tail -30 outputs/lora_1k_2gpu_r32_shaped_lr1e5/latest.log | grep -E 'format_reward|attempt_reward|correct_reward|val-core'"

# 3. Look for improvements:
# - format_reward increasing → model learning to use \boxed{}
# - attempt_reward increasing → model giving real answers
# - correct_reward appearing → model getting some right!
# - val-core accuracy > 15% → better than random
```

### Option B: Quick Smoke Test (2 minutes)

```bash
# Test shaped reward on 100 steps only
export CUDA_VISIBLE_DEVICES=6,7
export USE_SHAPED_REWARD=true

# Modify script temporarily
sed -i 's/trainer.total_epochs=10/trainer.total_epochs=1/g' run_memory_7B_lora_shaped_reward.sh
sed -i 's/data.train_batch_size=8/data.train_batch_size=4/g' run_memory_7B_lora_shaped_reward.sh

bash run_memory_7B_lora_shaped_reward.sh

# If format_reward > 0 in first 50 steps → shaped reward is working!
```

---

## Success Metrics

**After 250 steps, you should see:**
- ✅ format_reward: 0.1-0.3 (models use \boxed{} format)
- ✅ attempt_reward: occasional 0.2+ (models try to answer)
- ✅ Total score mean: 0.2-0.4 (was 0.0)

**After 500 steps:**
- ✅ format_reward: 0.3-0.5 consistently
- ✅ attempt_reward: 0.2-0.4 frequently
- ✅ correct_reward: occasional 0.5 (some correct answers!)
- ✅ Validation accuracy: 20-30%

**After 1250 steps:**
- ✅ Total score mean: 0.5-0.7
- ✅ Validation accuracy: 35-50%
- ✅ Training reward > 0 for 50%+ samples

---

## Troubleshooting

### Issue: format_reward still 0 after 100 steps
**Solution:** Model hasn't learned format. Try:
```bash
# Increase format reward weight
# In hotpotqa_shaped.py, change:
rewards['format_reward'] = 0.4  # Was 0.2
```

### Issue: attempt_reward stuck at 0
**Solution:** Model stuck on "no information" answers. Try:
```bash
# Add negative penalty for giving up
if 'no information' in answer or 'cannot' in answer:
    rewards['attempt_reward'] = -0.1  # Negative!
else:
    rewards['attempt_reward'] = 0.3
```

### Issue: Learning rate seems unstable
**Solution:** Reduce further
```bash
LEARNING_RATE=5e-6  # Even lower
```

### Issue: GPU OOM
**Solution:**
```bash
# Reduce batch size or tokens per GPU
data.train_batch_size=4 \
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12000 \
```

---

## Comparison: Binary vs Shaped Reward

### Binary Reward (Current)
```
Sample 1: answer="no info"  → reward=0.0
Sample 2: answer="no"       → reward=0.0
Sample 3: answer="\boxed{no}" → reward=0.0 (wrong answer)
Sample 4: answer="\boxed{yes}" → reward=1.0 (correct!)

Average reward: 0.25
Learning signal: Very sparse
```

### Shaped Reward (New)
```
Sample 1: answer="no info"
  → format=0, attempt=0, correct=0 → total=0.0

Sample 2: answer="no"
  → format=0.1 (fallback), attempt=0.2, correct=0 → total=0.3

Sample 3: answer="\boxed{no}"
  → format=0.2, attempt=0.3, correct=0 → total=0.5

Sample 4: answer="\boxed{yes}"
  → format=0.2, attempt=0.3, correct=0.5 → total=1.0

Average reward: 0.45
Learning signal: Much denser! Model gets feedback on partial progress
```

---

## Files Created

1. `/home/admin123/dl/MemAgent/verl/utils/reward_score/hotpotqa_shaped.py`
   - Shaped reward implementation

2. `/home/admin123/dl/MemAgent/run_memory_7B_lora_shaped_reward.sh`
   - Training script with all fixes

3. `/home/admin123/dl/MemAgent/verl/utils/reward_score/__init__.py`
   - Modified to support USE_SHAPED_REWARD env var

4. `/home/admin123/dl/MemAgent/verl/utils/reward_score/__init__.py.backup`
   - Backup of original file

---

## Recommended Next Steps

1. **Try shaped reward first** (easiest, most likely to help):
   ```bash
   bash run_memory_7B_lora_shaped_reward.sh
   ```

2. **If shaped reward helps but plateaus**, add curriculum learning

3. **If still struggling**, consider SFT pre-training

4. **If successful**, scale up to 3200 training samples

Good luck! 🚀
