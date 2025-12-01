# Memory Optimization Guide for 48GB GPU Training

## Problem Analysis

### Original OOM Error
- **Error Location**: `loss.backward()` in `verl/workers/actor/dp_actor.py:519`
- **Memory Usage**: 46.34 GB / 48 GB allocated by PyTorch
- **Root Cause**: Activation memory during backward pass exceeded available GPU memory

### Memory Breakdown (Original Configuration)

| Component | Memory Usage | Notes |
|-----------|-------------|-------|
| Actor Model (7B) | ~14 GB | bfloat16 parameters |
| Reference Model (7B) | ~14 GB | bfloat16 parameters |
| Critic Model (7B) | ~14 GB | bfloat16 parameters |
| Gradients | ~14 GB | During backward pass |
| Activations | ~6-8 GB | With gradient checkpointing, seq_len=512 |
| KV Cache | ~0.5 GB | kv_cache_max_length=800 |
| **Total** | **~46-48 GB** | **Exceeds 48GB during backward** |

## Optimization Strategy

### Three-Tier Approach

#### Tier 1: Standard Optimization (Recommended)
**Script**: `run_memory_7B_kv.sh` (already modified)

**Key Changes**:
```bash
MAXLEN=256              # Was: 512 (↓50%)
MAX_NEW_TOKEN=32        # Was: 40 (↓20%)
KV_CACHE_MAX_LEN=400    # Was: 800 (↓50%)
MAX_TOKEN_LEN_PER_GPU=400  # Was: 800 (↓50%)
chunk_size=128          # Was: 256 (↓50%)
```

**Mixed Precision**:
```bash
reduce_dtype=bfloat16   # Was: float32 (↓50% memory)
buffer_dtype=bfloat16   # Was: float32 (↓50% memory)
```

**Expected Memory**: ~32-35 GB (comfortable margin on 48GB GPU)

#### Tier 2: Minimal Configuration
**Script**: `run_memory_7B_kv_minimal.sh` (fallback if Tier 1 fails)

**Key Changes**:
```bash
MAXLEN=192              # Further reduced
MAX_NEW_TOKEN=24        # Further reduced
KV_CACHE_MAX_LEN=300    # Further reduced
MAX_TOKEN_LEN_PER_GPU=300  # Further reduced
chunk_size=96           # Further reduced
```

**Expected Memory**: ~26-30 GB (maximum safety margin)

#### Tier 3: Multi-GPU Configuration
If single GPU still fails, use 2 GPUs:
```bash
NGPUS_PER_NODE=2
CUDA_VISIBLE_DEVICES=0,1
actor_rollout_ref.actor.fsdp_config.fsdp_size=2
```

## Configuration Comparison

| Parameter | Original | Optimized | Minimal | Impact |
|-----------|----------|-----------|---------|--------|
| MAXLEN | 512 | 256 | 192 | High - Activation memory |
| KV_CACHE_MAX_LEN | 800 | 400 | 300 | Medium - KV cache size |
| MAX_TOKEN_LEN_PER_GPU | 800 | 400 | 300 | High - Total memory cap |
| chunk_size | 256 | 128 | 96 | Medium - Per-step memory |
| reduce_dtype | float32 | bfloat16 | bfloat16 | Medium - Gradient memory |
| buffer_dtype | float32 | bfloat16 | bfloat16 | Low - Buffer memory |

## Memory Saving Techniques Explained

### 1. Sequence Length Reduction
**Impact**: Most significant memory saving

- Activation memory scales quadratically with sequence length for attention: O(n²)
- Reducing MAXLEN from 512 to 256 saves ~50% activation memory
- Reducing to 192 saves ~64% activation memory

### 2. KV Cache Reduction
**Impact**: Direct linear memory saving

- Each token in KV cache consumes: `layers × heads × head_dim × 2(K,V) × dtype_size`
- For Qwen2.5-7B: 32 × 32 × 128 × 2 × 2 bytes = ~512 KB per token
- Reducing from 800 to 400 tokens saves ~200 MB per sample

### 3. Mixed Precision Optimization
**Impact**: Reduces gradient and intermediate tensor memory

- `reduce_dtype=bfloat16`: Gradient accumulation in bf16 instead of fp32 (50% saving)
- `buffer_dtype=bfloat16`: Communication buffers in bf16 (50% saving)
- Total saving: ~2-3 GB

### 4. CUDA Memory Allocator Tuning
**Impact**: Reduces memory fragmentation

```bash
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128,roundup_power2_divisions:16"
```

- `expandable_segments:True`: Allows memory pool to grow dynamically
- `max_split_size_mb:128`: Larger blocks reduce fragmentation
- `roundup_power2_divisions:16`: Better alignment for reuse

### 5. FSDP Offloading (Already Enabled)
**Impact**: Offloads parameters/optimizer to CPU when not in use

- `param_offload=True`: Parameters moved to CPU between forward/backward
- `optimizer_offload=True`: Optimizer states kept in CPU RAM
- `backward_prefetch=none`: No prefetching to minimize peak memory

## Usage Instructions

### Step 1: Try Standard Optimization
```bash
bash run_memory_7B_kv.sh
```

### Step 2: If Still OOM, Try Minimal
```bash
bash run_memory_7B_kv_minimal.sh
```

### Step 3: Monitor Memory
Watch GPU memory during training:
```bash
watch -n 1 nvidia-smi
```

### Step 4: Check for Memory Leaks
If OOM occurs after several iterations:
```bash
# Add to script before python command
export PYTORCH_NO_CUDA_MEMORY_CACHING=1  # Disable caching (slower but safer)
```

## Performance Trade-offs

### What You Lose
1. **Context Length**: Shorter sequences may lose some long-range context
2. **Throughput**: Smaller batches reduce GPU utilization
3. **Precision**: bfloat16 reduce has slightly lower numerical precision

### What You Keep
1. **Model Quality**: Shorter sequences still effective for most tasks
2. **Convergence**: PPO dynamics unchanged
3. **Final Performance**: Evaluation uses full model capacity

## Troubleshooting

### Still Getting OOM?

1. **Check GPU Usage**:
   ```bash
   nvidia-smi
   ```
   Ensure no other processes using the GPU

2. **Reduce Further**:
   - Try `MAXLEN=128`
   - Try `chunk_size=64`
   - Try `max_chunks=0` (no chunk limit)

3. **Use Gradient Accumulation**:
   ```bash
   actor_rollout_ref.actor.ppo_epochs=2  # More epochs, less memory per step
   ```

4. **Check Ray Workers**:
   ```bash
   ray status
   ```
   Ensure Ray isn't spawning extra workers

### Training Too Slow?

1. **Increase Sequence Length Gradually**:
   - Start with optimized config
   - Monitor memory
   - Increase MAXLEN by 64 until near limit

2. **Profile Memory**:
   ```python
   # Add to training code
   torch.cuda.memory._record_memory_history()
   ```

## Validation

### Expected Behavior
- **Training starts successfully**
- **Memory usage stays below 40 GB**
- **No OOM during backward pass**
- **Stable memory across iterations**

### Success Metrics
```
✓ Forward pass completes
✓ Backward pass completes
✓ Memory < 40GB peak
✓ No fragmentation warnings
```

## Additional Resources

- VERL Documentation: [verl/docs/](verl/docs/)
- FSDP Memory Guide: https://pytorch.org/docs/stable/fsdp.html
- KV Cache Implementation: [recurrent/impls/kvcache_memory.py](recurrent/impls/kvcache_memory.py:1)

## Contact

If issues persist, please check:
1. GPU model and available memory: `nvidia-smi`
2. PyTorch version: `python -c "import torch; print(torch.__version__)"`
3. Ray cluster status: `ray status`
4. System memory: `free -h`
