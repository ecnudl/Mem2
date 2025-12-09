# LoRA PPO Training Implementation Plan

## Executive Summary

Based on analysis of your 7B full fine-tuning results showing instability (F1 scores fluctuating between 19-28%), **LoRA training is highly recommended** for your scenario. This document provides three implementation approaches.

## Background: Why LoRA for Your Case

### Current Training Issues
- **Instability**: Performance peaks at step_2000 (F1=28.28) then degrades
- **Overfitting**: Small dataset (1k samples) + large model (7B) = high risk
- **Resource intensive**: 4-GPU FSDP required, high memory usage
- **RL volatility**: PPO training inherently unstable, amplified by full fine-tuning

### LoRA Benefits
1. **Stability**: Only trains 0.1-1% of parameters → more stable gradients
2. **Anti-overfitting**: Fewer trainable params → better generalization
3. **Efficiency**: 90% less memory → potentially single-GPU training
4. **Faster convergence**: Higher learning rates possible (1e-4 vs 1e-6)
5. **Better for RL**: Constrains policy updates, reduces divergence

## Implementation Approaches

### **Approach A: Add LoRA to PPO Actor (Recommended)**

**Pros**:
- Direct solution to the problem
- Full LoRA benefits in RL training
- Integrated into framework

**Cons**:
- Requires modifying core VERL code
- ~200 lines of code changes

**Estimated time**: 2-4 hours implementation + testing

---

### **Approach B: LoRA SFT → Full PPO (Quick Start)**

**Pros**:
- No PPO code modification needed
- Can start immediately
- Validated approach (SFT trainer has LoRA)

**Cons**:
- PPO stage still uses full fine-tuning
- Two-stage training complexity

**Estimated time**: 30 minutes setup

---

### **Approach C: Continue LoRA Throughout (Hybrid)**

**Pros**:
- LoRA benefits throughout pipeline
- Most parameter-efficient

**Cons**:
- Requires both modifications
- Complex checkpoint management

**Estimated time**: 4-6 hours

---

## Detailed Implementation: Approach A (Recommended)

### Step 1: Modify FSDP Worker to Support LoRA

**File**: `verl/workers/fsdp_workers.py`

**Location**: `_build_model_optimizer` method (around line 230)

**Add this code after model initialization**:

```python
# After line 230 (after applying Liger kernel)
if use_liger:
    from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
    _apply_liger_kernel_to_instance(model=actor_module)

# ADD: LoRA support
lora_rank = self.config.model.get("lora_rank", 0)
if lora_rank > 0:
    from peft import LoraConfig, TaskType, get_peft_model

    actor_module.enable_input_require_grads()

    lora_config = {
        "task_type": TaskType.CAUSAL_LM,
        "r": lora_rank,
        "lora_alpha": self.config.model.get("lora_alpha", 16),
        "target_modules": self.config.model.get("target_modules", "all-linear"),
        "lora_dropout": self.config.model.get("lora_dropout", 0.0),
        "bias": "none",
    }

    if self.rank == 0:
        print(f"[{role}] Applying LoRA with config: {lora_config}")

    actor_module = get_peft_model(actor_module, LoraConfig(**lora_config))

    if self.rank == 0:
        actor_module.print_trainable_parameters()

# Continue with existing code...
actor_module.to(torch_dtype)
```

**Location**: Update FSDP wrap policy (around line 274)

```python
auto_wrap_policy = get_fsdp_wrap_policy(
    module=actor_module,
    config=fsdp_config.get("wrap_policy", None),
    is_lora=lora_rank > 0  # ADD: Pass LoRA flag
)
```

### Step 2: Update Configuration Schema

**File**: `verl/trainer/config/ppo_trainer.yaml`

**Location**: `actor_rollout_ref.model` section (around line 24)

```yaml
actor_rollout_ref:
  model:
    path: ~/models/deepseek-llm-7b-chat
    external_lib: null
    override_config: { }
    enable_gradient_checkpointing: True
    use_remove_padding: False
    use_liger: False
    # ADD: LoRA configuration
    lora_rank: 0  # Set to positive value to enable LoRA (e.g., 32, 64)
    lora_alpha: 16  # LoRA scaling factor (typically 16 or 32)
    lora_dropout: 0.0  # LoRA dropout (typically 0.0 for PPO)
    target_modules: all-linear  # or [q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj]
```

### Step 3: Update `_build_model_optimizer` signature

**File**: `verl/workers/fsdp_workers.py`

**Location**: Method signature (around line 152)

```python
def _build_model_optimizer(
    self,
    model_path,
    fsdp_config,
    optim_config,
    override_model_config,
    use_remove_padding=False,
    enable_gradient_checkpointing=False,
    trust_remote_code=False,
    use_liger=False,
    lora_rank=0,  # ADD
    lora_alpha=16,  # ADD
    lora_dropout=0.0,  # ADD
    target_modules="all-linear",  # ADD
    role="actor",
):
```

**Update call sites** (around line 350 for actor, line 450 for ref):

```python
# For actor
self.actor_module, self.actor_optimizer = self._build_model_optimizer(
    model_path=self.config.model.path,
    fsdp_config=self.config.actor.fsdp_config,
    optim_config=self.config.actor.optim,
    override_model_config=self.config.model.override_config,
    use_remove_padding=self.config.model.use_remove_padding,
    enable_gradient_checkpointing=self.config.model.enable_gradient_checkpointing,
    trust_remote_code=self.config.model.get("trust_remote_code", False),
    use_liger=self.config.model.get("use_liger", False),
    lora_rank=self.config.model.get("lora_rank", 0),  # ADD
    lora_alpha=self.config.model.get("lora_alpha", 16),  # ADD
    lora_dropout=self.config.model.get("lora_dropout", 0.0),  # ADD
    target_modules=self.config.model.get("target_modules", "all-linear"),  # ADD
    role="actor",
)
```

### Step 4: Update FSDP Wrap Policy

**File**: `verl/utils/fsdp_utils.py`

**Find function**: `get_fsdp_wrap_policy`

**Update signature and logic**:

```python
def get_fsdp_wrap_policy(
    module: nn.Module,
    config: DictConfig = None,
    is_lora: bool = False,  # ADD
):
    """
    Get FSDP auto wrap policy.

    Args:
        module: The model module
        config: Wrap policy configuration
        is_lora: Whether LoRA is enabled (affects wrapping strategy)
    """
    if is_lora:
        # For LoRA, we need to ensure LoRA layers are wrapped correctly
        from peft.tuners.lora import LoraLayer

        def lambda_policy_lora(module):
            # Wrap LoRA layers and original transformer layers
            if isinstance(module, LoraLayer):
                return True
            return lambda_policy_fn(module)  # Use default policy for base layers

        return lambda_policy_lora
    else:
        # Original logic
        return lambda_policy_fn
```

### Step 5: Create LoRA Training Script

**File**: `run_memory_7B_lora.sh`

```bash
#!/bin/bash
set -x

# 7B LoRA Training Script (Single GPU or 2-GPU)

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate memagent

# Environment setup
export RAY_ADDRESS=""
unset RAY_ADDRESS 2>/dev/null || true
export RAY_TMPDIR=/home/admin123/dl/MemAgent/outputs/ray_tmp
mkdir -p "$RAY_TMPDIR"

# Use 1 or 2 GPUs for LoRA training (much less memory needed)
export CUDA_VISIBLE_DEVICES=0,1  # Can even use single GPU: CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Configuration
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-2}  # LoRA can work with 2 GPUs or even 1
FSDP_SIZE=${FSDP_SIZE:-$((NNODES * NGPUS_PER_NODE))}

PROJ_ROOT=/home/admin123/dl/MemAgent/outputs
DATASET_ROOT=/home/admin123/dl/MemAgent/taskutils/memory_data

MODEL_PATH=/mnt/ssd2/models/Qwen2.5-7B-Instruct
VAL_PATH="${DATASET_ROOT}/hotpotqa/hotpotqa_dev_20.parquet"
TRAIN_PATH="${DATASET_ROOT}/hotpotqa/hotpotqa_train_1k.parquet"
EXP=lora_1k_2gpu_r64
PROJ_DIR=${PROJ_ROOT}/${EXP}

MAXLEN=256
MAX_NEW_TOKEN=64

# LoRA allows higher learning rate and faster convergence
LEARNING_RATE=5e-5  # 50x higher than full fine-tuning!

python3 -c "import ray; ray.shutdown()" 2>/dev/null || true
export RAY_DISABLE_IMPORT_WARNING=1

python3 -m verl.trainer.main_ppo \
    recurrent.enable=memory \
    recurrent.memory.config.chunk_size=1536 \
    recurrent.memory.config.max_chunks=16 \
    recurrent.memory.config.max_memorization_length=${MAX_NEW_TOKEN} \
    recurrent.memory.config.max_final_response_length=${MAX_NEW_TOKEN} \
    algorithm.adv_estimator=grpo \
    algorithm.grpo_use_adv=False \
    trainer.save_freq=1000 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    trainer.logger=['console'] \
    actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
    actor_rollout_ref.actor.clip_ratio_high=0.20 \
    actor_rollout_ref.actor.entropy_coeff=0.000 \
    data.train_files=$TRAIN_PATH \
    data.val_files=$VAL_PATH \
    data.shuffle=False \
    data.filter_overlong_prompts=True \
    data.train_batch_size=4 \
    data.truncation='center' \
    +data.context_key='context' \
    data.max_prompt_length=$MAXLEN \
    data.max_response_length=$MAX_NEW_TOKEN \
    reward_model.reward_manager='naive' \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    actor_rollout_ref.actor.optim.lr=${LEARNING_RATE} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16000 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16000 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16000 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=none \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.load_format=dummy_dtensor \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${FSDP_SIZE} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${NGPUS_PER_NODE} \
    actor_rollout_ref.rollout.max_model_len=1024 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.critic_warmup=0 \
    critic.model.enable_gradient_checkpointing=True \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.lora_dropout=0.0 \
    actor_rollout_ref.model.target_modules=all-linear \
    trainer.project_name='verl-hongli' \
    trainer.experiment_name=${EXP} \
    trainer.val_before_train=False \
    trainer.resume_mode=auto \
    trainer.resume_from_path=null \
    trainer.n_gpus_per_node=$NGPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.test_freq=20 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$PROJ_DIR \
    trainer.total_epochs=10
```

### Step 6: LoRA Hyperparameter Recommendations

Based on literature and your 7B model:

```yaml
# Conservative (more stable, recommended for first try)
lora_rank: 32
lora_alpha: 16
learning_rate: 2e-5

# Balanced (good tradeoff)
lora_rank: 64
lora_alpha: 32
learning_rate: 5e-5

# Aggressive (faster learning, higher capacity)
lora_rank: 128
lora_alpha: 64
learning_rate: 1e-4
```

**Target modules for Qwen2.5**:
```yaml
# All linear layers (highest capacity, recommended)
target_modules: all-linear

# Attention only (parameter-efficient)
target_modules: [q_proj, k_proj, v_proj, o_proj]

# Attention + FFN (balanced)
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

---

## Implementation: Approach B (Quick Start)

### Step 1: LoRA SFT Pre-training

Create `sft_7B_lora.sh`:

```bash
#!/bin/bash
set -x

nproc_per_node=2  # Use 2 GPUs for 7B LoRA SFT
save_path=/home/admin123/dl/MemAgent/outputs/sft_7B_lora

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/home/admin123/dl/MemAgent/taskutils/memory_data/hotpotqa/hotpotqa_train_1k.parquet \
    data.val_files=/home/admin123/dl/MemAgent/taskutils/memory_data/hotpotqa/hotpotqa_dev_20.parquet \
    data.prompt_key=prompt \
    data.response_key=response \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=/mnt/ssd2/models/Qwen2.5-7B-Instruct \
    trainer.default_local_dir=$save_path \
    trainer.project_name=memagent-sft \
    trainer.experiment_name=7B-lora-sft \
    trainer.logger=['console'] \
    trainer.total_epochs=3 \
    trainer.default_hdfs_dir=null \
    model.lora_rank=64 \
    model.lora_alpha=32 \
    model.target_modules=all-linear \
    optim.lr=1e-4
```

### Step 2: Merge LoRA Weights

After SFT, merge LoRA back to base model:

```python
# merge_lora.py
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_path = "/mnt/ssd2/models/Qwen2.5-7B-Instruct"
lora_checkpoint = "/home/admin123/dl/MemAgent/outputs/sft_7B_lora/checkpoint_final"
output_path = "/home/admin123/dl/MemAgent/outputs/sft_7B_lora_merged"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype="bfloat16",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, lora_checkpoint)
merged_model = model.merge_and_unload()

merged_model.save_pretrained(output_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained(output_path)

print(f"Merged model saved to {output_path}")
```

### Step 3: Continue with PPO

Use merged model as starting point for PPO:

```bash
# In 7Bnodes.sh, update:
MODEL_PATH=/home/admin123/dl/MemAgent/outputs/sft_7B_lora_merged
```

---

## Testing Plan

### Phase 1: Validation (Estimated 2-4 hours)

1. **Smoke test LoRA initialization**:
   ```bash
   # Run for 10 steps to verify LoRA loads
   python3 -m verl.trainer.main_ppo <config> trainer.total_epochs=0.01
   ```

2. **Check trainable parameters**:
   - Full fine-tuning: ~7B parameters
   - LoRA (r=64): ~40-80M parameters (~1% of full model)

3. **Memory usage comparison**:
   ```bash
   # Monitor GPU memory
   nvidia-smi dmon -s mu
   ```
   - Expected: 30-50% less memory than full fine-tuning

### Phase 2: Short Training Run (Estimated 4-8 hours)

1. **Train for 2000 steps**:
   ```bash
   bash run_memory_7B_lora.sh
   ```

2. **Compare metrics to full fine-tuning**:
   - Training stability (loss curves)
   - Validation EM/F1 at step 2000
   - GPU memory peak usage

### Phase 3: Full Training (Estimated 1-2 days)

1. **Complete 10 epochs**
2. **Evaluate all checkpoints**
3. **Compare final results**

---

## Expected Results

### Memory Reduction
- **Full fine-tuning**: ~45GB per GPU (4 GPUs required)
- **LoRA (r=64)**: ~25-30GB per GPU (2 GPUs sufficient)
- **Saving**: 40-50% memory reduction

### Training Stability
- **Full fine-tuning**: High variance in metrics (your current results)
- **LoRA**: Expected 30-50% reduction in variance
- **Convergence**: Faster initial learning, more stable plateau

### Performance Target
- **Current best (full)**: F1=28.28 at step_2000, then degrades
- **LoRA target**: F1=28-32 at step_2000, maintains performance
- **Key metric**: Consistency across steps 2000-5000

---

## Rollback Plan

If LoRA underperforms:

1. **Keep full fine-tuning code intact** (don't delete `7Bnodes.sh`)
2. **LoRA modifications are additive** (can be disabled with `lora_rank=0`)
3. **Quick rollback**:
   ```bash
   # Disable LoRA in config
   actor_rollout_ref.model.lora_rank=0
   ```

---

## Troubleshooting

### Issue: "PEFT not installed"
```bash
pip install peft
```

### Issue: "LoRA layers not wrapped correctly in FSDP"
- Check `get_fsdp_wrap_policy` includes `LoraLayer`
- Verify `use_orig_params=True` in FSDP config

### Issue: "Learning rate too high, divergence"
- LoRA can use higher LR, but start conservative: `lr=2e-5`
- Gradually increase to `5e-5` or `1e-4` if stable

### Issue: "Checkpoint loading fails"
- LoRA checkpoints save adapter weights separately
- Use PEFT's `PeftModel.from_pretrained()` for loading

---

## Next Steps

Choose your approach:

1. **If you want best results**: Implement Approach A (2-4 hours coding + testing)
2. **If you want quick validation**: Implement Approach B (30 min setup + overnight SFT)
3. **If you want to experiment first**: Start with Approach B, then migrate to A

I recommend **starting with Approach B** to validate LoRA helps, then implementing Approach A for long-term use.

Would you like me to:
1. Generate the code patches for Approach A?
2. Create the quick-start scripts for Approach B?
3. Both?
