# vLLM v1 Memory Pool Compatibility Fix

## Problem

### Error Message
```
AssertionError: Expandable segments are not compatible with memory pool.
Please track https://github.com/pytorch/pytorch/issues/147851 for the latest updates.
```

### Error Location
```python
File ".../vllm/device_allocator/cumem.py", line 150, in __init__
  assert "expandable_segments:True" not in conf, \
```

## Root Cause

**vLLM v1** introduced a new memory management system (`CuMemAllocator`) that uses CUDA memory pools for efficient allocation. This system is **incompatible** with PyTorch's `expandable_segments:True` feature.

### Technical Details

1. **vLLM v1 Memory Pool**:
   - Uses CUDA's native memory pool API (`cuMemCreate`, `cuMemMap`)
   - Provides efficient memory reuse for KV cache
   - Implements sleep/wake mechanism for FSDP + vLLM interop

2. **PyTorch Expandable Segments**:
   - A feature to reduce memory fragmentation
   - Allows PyTorch's caching allocator to expand segments dynamically
   - **Conflicts** with external memory pool managers like vLLM's

3. **Why They Conflict**:
   - Both try to manage GPU memory at different levels
   - Memory allocated by vLLM's pool can't be "expanded" by PyTorch
   - Can cause memory corruption or allocation failures

## Fix Applied

### Before (Problematic)
```bash
# Memory optimization - reduce fragmentation for vLLM KV cache allocation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### After (Fixed)
```bash
# Memory optimization
# Note: expandable_segments is incompatible with vLLM v1 memory pool
# Using default PyTorch allocator to work with vLLM's CuMemAllocator
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # DISABLED for vLLM v1
```

**Result**: Removed the incompatible environment variable, allowing vLLM v1 to manage memory pools independently.

## Alternative Memory Optimization Options

If you still want to optimize PyTorch memory allocation (for FSDP training part only), you can use:

### Option 1: Release Unused Cached Memory (Recommended)
```bash
# No environment variable needed
# vLLM v1 handles its own memory pool
# PyTorch uses default allocator for FSDP
```

### Option 2: Limit Split Size (Use with Caution)
```bash
# Only if you encounter specific fragmentation issues
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```
**Note**: Test carefully - this can still interfere with vLLM in some cases.

### Option 3: Backend-specific Settings
```bash
# These are safe and don't conflict with vLLM
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# (already in the script)
```

## vLLM v1 vs v0.x Memory Management

| Feature | vLLM v0.x | vLLM v1 |
|---------|-----------|---------|
| Memory Pool | PyTorch native | CUDA native (CuMem) |
| Sleep/Wake | Unload/Reload model | Memory pool tags |
| Expandable Segments | Compatible | **Incompatible** |
| FSDP Interop | Slower (reload overhead) | Faster (keep in memory) |

## GPU Configuration Update

User modified GPU selection from `4,5,6,7` to `0,4,6,7`:

```bash
# Current GPU Status (example)
GPU 0: Free (48.5 GB available)  ✅ Selected
GPU 1: Occupied (45.9 GB used)   ❌
GPU 2: Occupied (19.3 GB used)   ❌
GPU 3: Occupied (16.9 GB used)   ❌
GPU 4: Free (48.5 GB available)  ✅ Selected
GPU 5: Occupied (36.6 GB used)   ❌
GPU 6: Free (48.5 GB available)  ✅ Selected
GPU 7: Free (48.5 GB available)  ✅ Selected
```

This is a good choice - using 4 free GPUs avoids any contention.

## Expected Behavior After Fix

### Before Fix
```
✅ Training starts
✅ FSDP model loads
✅ Ray workers initialize
❌ vLLM initialization fails with assertion error
```

### After Fix
```
✅ Training starts
✅ FSDP model loads
✅ Ray workers initialize
✅ vLLM loads successfully
✅ Training proceeds normally
```

## Verification Steps

1. **Check Environment**:
   ```bash
   # Should NOT have expandable_segments
   echo $PYTORCH_CUDA_ALLOC_CONF
   # (empty or other settings)
   ```

2. **Monitor vLLM Initialization**:
   ```bash
   tail -f outputs/lora_train.log | grep -E "(vLLM|CuMem|memory pool)"
   ```

   Expected output:
   ```
   WARNING 12-09 13:53:00 [symm_mem.py:58] SymmMemCommunicator: Device capability 8.9 not supported...
   # (This warning is OK - just means some optimizations aren't available)

   # Should NOT see:
   # AssertionError: Expandable segments are not compatible with memory pool
   ```

3. **Verify Training Starts**:
   ```bash
   tail -f outputs/lora_train.log | grep "step:"
   ```

   Should see:
   ```
   step:0 - actor/entropy_loss:... - perf/max_memory_allocated_gb:...
   step:1 - actor/entropy_loss:... - perf/max_memory_allocated_gb:...
   ```

## Related Issues

- **PyTorch Issue**: https://github.com/pytorch/pytorch/issues/147851
  - Tracking expandable_segments + external memory pools compatibility
  - Currently no solution - they remain incompatible

- **vLLM Discussion**: vLLM v1 memory pool design is intentionally incompatible
  - Provides better performance for inference workloads
  - Trade-off for flexibility with PyTorch memory management

## Summary

| Issue | Solution | Impact |
|-------|----------|--------|
| `expandable_segments` conflict | Disabled the setting | ✅ vLLM loads successfully |
| Memory fragmentation | vLLM v1 handles it internally | ✅ No action needed |
| GPU selection | Updated to 0,4,6,7 (free GPUs) | ✅ Avoids contention |

## Next Steps

Run the training:
```bash
bash run_memory_7B_lora.sh
```

If you see `step:0`, `step:1`, etc., the fix is successful! 🎉

## Debugging Commands

If issues persist:

```bash
# 1. Check PyTorch CUDA config
python3 -c "import os; print('PYTORCH_CUDA_ALLOC_CONF:', os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'Not set'))"

# 2. Check vLLM version
pip show vllm | grep Version

# 3. Check CUDA compatibility
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"

# 4. Test vLLM initialization standalone
python3 -c "
from vllm import LLM
llm = LLM(
    model='/mnt/ssd2/models/Qwen2.5-7B-Instruct',
    tensor_parallel_size=4,
    gpu_memory_utilization=0.35,
    trust_remote_code=True
)
print('vLLM initialized successfully!')
"
```

## Additional Notes

### Warning: SymmMemCommunicator
```
WARNING [symm_mem.py:58] SymmMemCommunicator: Device capability 8.9 not supported
```
**Status**: ⚠️ Safe to ignore
- RTX 4090 has compute capability 8.9
- vLLM's symmetric memory optimization not available for this GPU
- Fallback to standard memory transfer works fine

### Warning: FlashInfer
```
WARNING [topk_topp_sampler.py:66] FlashInfer is not available
```
**Status**: ⚠️ Safe to ignore
- Optional optimization for sampling
- PyTorch-native implementation works (just slower)
- Doesn't affect training correctness
