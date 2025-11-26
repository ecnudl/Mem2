# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MemAgent is a reinforcement learning framework for training long-context LLMs using a novel memory agent architecture. The system enables models to process arbitrarily long texts within fixed context windows through multi-conversation RL-based memory mechanisms. Built on the VERL framework with custom recurrent agent extensions.

## Core Architecture

### Training Pipeline
The training system follows a distributed PPO architecture coordinated by Ray:

1. **Entry Point**: `verl/trainer/main_ppo.py` - Initializes Ray cluster and launches RayPPOTrainer
2. **Trainer**: `verl/trainer/ppo/ray_trainer.py` - Orchestrates rollout, reward computation, and PPO updates
3. **Workers**: FSDP-based actor, critic, and reward workers distributed across Ray actors
4. **Memory Agents**: `recurrent/impls/memory.py` and `recurrent/impls/kvcache_memory.py` implement the core memory mechanisms

### Memory Agent Modes

**Text Memory Mode** (`recurrent/impls/memory.py`):
- Stores memory as text token sequences
- Re-encodes entire prompt + memory + chunk each turn
- Simpler implementation but less efficient

**KV Cache Mode** (`recurrent/impls/kvcache_memory.py`):
- Stores memory as past_key_values tensors
- Uses prefill/decode stages to avoid re-encoding
- More memory and compute efficient for long contexts

Key utilities for KV cache: `recurrent/kvcache_utils.py` (concat_past_kv, truncate_past_kv, kv_seq_len)

### Recurrent Agent Interface

All memory agents must implement the `RAgent` interface defined in `recurrent/interface.py`:
- `start(gen_batch, timing_raw)`: Initialize agent state for a new batch
- `action()`: Construct input prompts and metadata for next generation step
- `update(gen_output)`: Process model output and update memory state
- `done()`: Return True when generation should terminate
- `end()`: Cleanup and return final_mask/sample_index for reward computation

Async agents implement `AsyncRAgent` for OpenAI-style API-based rollout.

### Data Flow

1. **Dataset**: `RDataset` subclasses load parquet files with context/prompt fields
2. **Generation Manager**: `recurrent/generation_manager.py` runs the multi-turn loop calling action/update/done
3. **Rollout Workers**: `verl/workers/rollout/` handle tokenization and LLM inference (HF or vLLM backends)
4. **Reward Computation**: Verifier functions in `verl/utils/reward_score/` or `taskutils/memory_data/hotpotqa_verifier.py`

## Common Commands

### Training

**7B Model (Single Node)**:
```bash
bash run_memory_7B.sh
```
Configure `PROJ_ROOT`, `DATASET_ROOT`, and `MODEL_PATH` inside the script first.

**14B Model (Multi-Node)**:
```bash
bash run_memory_14B.sh
```
Requires at least 2 nodes with 8 GPUs each. Set up Ray cluster before running.

**Smoke Test**:
```bash
bash run_memory_7B_smoke.sh
```
Quick GPU sanity check for recurrent agent.

### Evaluation

**Prepare Test Data**:
```bash
cd taskutils/memory_data
bash download_qa_dataset.sh
cd ../..
export DATAROOT=$(pwd)/hotpotqa
```

**Run Tests**:
```bash
cd taskutils/memory_eval
python run.py
```
Uses all available GPUs via Ray serve. Set `SERVE_PORT` and `DASH_PORT` if using Ray cluster.

**Inference Quickstart**:
```bash
# Local vLLM deployment
vllm serve BytedTsinghua-SIA/RL-MemoryAgent-14B --tensor_parallel_size 2
python quickstart.py --model BytedTsinghua-SIA/RL-MemoryAgent-14B

# Online API
export URL=https://your-endpoint
export API_KEY=your-key
python quickstart.py --model gpt-4o-2024-11-20
```

### Data Preparation

```bash
cd taskutils/memory_data
pip install nltk pyyaml beautifulsoup4 html2text wonderwords tenacity fire

# Process train/dev splits
python processing.py

# Filter with deployed Qwen models
python filter.py -i hotpotqa_dev_process.parquet -o hotpotqa_dev_result --noresume
python filter2.py

# Create eval datasets
export DATAROOT=/path/to/hotpotqa_dev.parquet
python convert_to_eval.py
python different_docs_eval.py
```

### Testing
```bash
# Fast regression tests
python -m pytest tests/sanity tests/utility

# Full test suite (requires GPUs)
python -m pytest tests/
```

## Key Configuration Patterns

### Switching Memory Modes

**Text Memory** (default in `run_memory_7B.sh`):
```bash
recurrent.enable=memory \
recurrent.memory.config.chunk_size=3000
```

**KV Cache Memory** (see `run_memory_7B_kv.sh`):
```bash
recurrent.enable=memory \
recurrent.memory.path=/path/to/recurrent/impls/kvcache_memory.py \
recurrent.memory.name=REGISTER \
recurrent.memory.config.chunk_size=1024 \
recurrent.memory.config.kv_cache_max_length=4096 \
recurrent.memory.config.reuse_prefill=True
```

### Rollout Backend Selection

- `actor_rollout_ref.rollout.name=hf`: Use HuggingFace transformers (single GPU, slower)
- `actor_rollout_ref.rollout.name=vllm`: Use vLLM (multi-GPU, faster, requires vLLM 0.5.4+)

For KV cache mode, ensure the rollout backend supports `past_key_values` (check `verl/workers/rollout/naive/naive_rollout_kv.py`).

### Memory and Performance Tuning

- `actor_rollout_ref.rollout.gpu_memory_utilization`: Fraction of GPU memory for vLLM KV cache (default 0.4-0.6)
- `actor_rollout_ref.actor.ppo_max_token_len_per_gpu`: Max tokens per GPU for actor training batches
- `actor_rollout_ref.actor.use_dynamic_bsz=True`: Enable dynamic batch sizing based on token count
- `actor_rollout_ref.actor.fsdp_config.param_offload=True`: Offload parameters to CPU to save GPU memory

## Important Implementation Details

### Reward Verification

Training uses a **stricter verifier** than testing:
- Training (`verl/utils/reward_score/hotpotqa.py`): Requires exact `\boxed{}` format with case matching
- Testing (`taskutils/memory_eval/utils/__init__.py`): Ignores articles, case, and punctuation

This gap is intentional to prevent reward hacking. Expect ~50% validation accuracy during training but ~80%+ at test time with the relaxed verifier.

### Qwen Model Configuration

Qwen2.5-Instruct models require manual YaRN activation for long context:
```bash
bash hfd.sh Qwen/Qwen2.5-7B-Instruct --tool aria2c -x 10
# Edit config.json to enable YaRN following instructions in model repo
export MODELROOT=/your/path/to/models
mv Qwen2.5-7B-Instruct $MODELROOT/Qwen2.5-7B-Instruct-128K
```

### Chat Template Modifications

The framework modifies `chat_template` to support tool-response masking without tensor operations. See `recurrent/chat_template/` for template utilities. Templates must support:
- Tool/observation turns for multi-turn workflows
- Context-independent multi-conversation format (agent outputs list of conversation lists)

### Ray Process Pool

CPU-intensive tasks (reward computation, tool calling) run in RayActor process pools to avoid blocking the head node. See `verl/workers/reward_manager/thread.py`. GPU tasks (LLM generation) also submit asynchronously to Ray.

## Environment Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install httpx==0.23.1 aiohttp -U ray[serve,default] vllm
pip install -r requirements.txt  # Installs VERL, Hydra, FlashAttention, transformers
```

For development, install pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```

## File Organization

- `verl/`: Core PPO trainer, workers, utilities (based on VERL framework)
- `recurrent/`: Memory agent interface, implementations, and generation managers
- `taskutils/memory_data/`: Dataset processing, synthetic data generation, verifiers
- `taskutils/memory_eval/`: Evaluation harnesses for HotpotQA and RULER tasks
- `run_memory_*.sh`: Training launch scripts with hyperparameters
- `quickstart.py`: Standalone inference script for deployed models

## Debugging Tips

- Add logging in `MemoryAgent.update()` or `KVCacheMemoryAgent.update()` to inspect memory accumulation
- Check `past_key_values` shapes in KV cache mode - common source of dimension mismatches
- Monitor `kv_cache_max_length` to avoid OOM - KV cache consumes significant GPU memory
- Verify rollout mode supports KV cache if switching from text memory (`verl/workers/fsdp_workers.py` → `_build_rollout`)
- Use `trainer.logger=['console']` for local debugging instead of wandb
