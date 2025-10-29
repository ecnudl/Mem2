# Repository Guidelines

## Project Structure & Module Organization
MemAgent pairs reinforcement-learning trainers with long-context inference tooling. Core training code lives in `verl/` (Ray/Hydra PPO trainers, reward utilities) and `recurrent/` (recurrent attention helpers, chat templates). Serving clients and Ray runtime config live in `serve/`. Evaluation harnesses and dataset scripts reside under `taskutils/memory_eval/` and `taskutils/memory_data/`; run scripts and checkpoints sit in `run_memory_*.sh` and `tests/`. Standalone utilities are in `scripts/` and `quickstart.py`. Dockerfiles in `docker/` reproduce the expected CUDA/vLLM environment.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: create an isolated Python 3.10+ runtime.
- `pip install -r requirements.txt`: install core trainers, Ray, Hydra, FlashAttention, and pre-commit hooks.
- `python quickstart.py --model BytedTsinghua-SIA/RL-MemoryAgent-14B`: run the memory agent loop against a local or remote endpoint.
- `python -m pytest tests/sanity tests/utility`: execute fast regression suites before touching heavier GPU tests.
- `bash run_memory_7B_smoke.sh`: launch a GPU smoke test that exercises the recurrent agent end-to-end.

## Coding Style & Naming Conventions
Use 4-space indentation and PEP 8 imports (stdlib, third-party, local). Mirror existing modules by typing public functions (`def run_ppo(config) -> None`) and raising explicit errors. Keep docstrings concise; add bilingual comments only when they clarify hardware or deployment assumptions. Files under `recurrent/` and `verl/` expect snake_case modules, CamelCase classes, and imperative verb function names.

## Testing Guidelines
Pytest is the canonical harness. Place unit tests alongside mirrored package folders in `tests/` (e.g., new Ray utilities go under `tests/verl/utils`). Name test files `test_<feature>.py` and parametrize heavy cases to gate optional GPU coverage. Always run at least the targeted sub-suite you touched plus `pytest tests/sanity` before opening a PR, and document any skipped long-running suites (`run_memory_14B.sh`, `taskutils/memory_eval/run.py`).

## Commit & Pull Request Guidelines
Recent history favours short, imperative subject lines (`add RL 7B`, `Fix typo`). Group related changes into a single commit and avoid merging generated checkpoints. Pull requests should outline motivation, list executed commands (tests, scripts), note required GPUs or datasets, and attach screenshots or logs for pipelines such as `quickstart.py` outputs.

## Agent Response Guidelines
All outbound communications must be delivered in Chinese, using a professional and concise voice.

## Security & Configuration Tips
Never commit real API keys—the default in `quickstart.py` is a placeholder. Store endpoints in environment variables (`export URL=...`, `export API_KEY=...`) or `.env` files ignored by Git. When sharing configs, scrub absolute paths and customer data from Ray or Hydra YAML before publishing.
