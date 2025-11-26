#!/usr/bin/env python3
"""
Ray集群版本冲突的临时解决方案包装脚本
使用独立的Ray实例，不影响其他GPU上的进程
"""
import os
import sys
import tempfile

# 创建独立的临时目录
temp_dir = os.environ.get("RAY_TMPDIR", f"/tmp/ray_kv_test_{os.getpid()}")
os.makedirs(temp_dir, exist_ok=True)

# 设置环境变量
os.environ["RAY_TMPDIR"] = temp_dir
os.environ["RAY_ADDRESS"] = ""

import hydra
import ray
from omegaconf import OmegaConf

# 导入原始main_ppo_kv的TaskRunner和其他函数
from verl.trainer.main_ppo_kv import TaskRunner, get_custom_reward_fn


@hydra.main(config_path="verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    """修改版run_ppo，使用独立Ray集群"""
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    # 关键修改：使用_temp_dir和ignore_reinit_error避免版本冲突
    if not ray.is_initialized():
        temp_dir = os.environ.get("RAY_TMPDIR")
        print(f"[Ray Init] 使用独立集群，临时目录: {temp_dir}")

        # 获取当前conda环境路径
        conda_env = os.environ.get("CONDA_PREFIX", "")
        python_path = os.path.join(conda_env, "bin/python") if conda_env else sys.executable

        print(f"[Ray Init] 使用Python环境: {python_path}")

        ray.init(
            _temp_dir=temp_dir,  # 使用独立临时目录
            ignore_reinit_error=True,  # 忽略重新初始化错误
            include_dashboard=False,  # 关闭dashboard避免端口冲突
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "WARN",
                    "PATH": os.environ.get("PATH", ""),
                    "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
                    "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
                },
                "worker_process_setup_hook": lambda: None,
            },
            num_cpus=config.ray_init.num_cpus,
        )
        print(f"[Ray Init] 成功启动 Ray {ray.__version__}")

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


if __name__ == "__main__":
    main()
