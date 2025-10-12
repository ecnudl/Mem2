# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os  # 操作系统环境变量与文件工具

import hydra  # Hydra 配置解析与管理框架
import ray  # Ray 分布式计算框架

from verl.trainer.ppo.ray_trainer import RayPPOTrainer  # PPO 主训练器
from verl.trainer.ppo.reward import load_reward_manager  # 奖励函数加载工具
import uvloop  # 高性能事件循环库
uvloop.install()  # 将 uvloop 设置为默认事件循环，提升异步性能


def get_custom_reward_fn(config):  # 根据配置动态加载自定义奖励函数
    import importlib.util  # 用于按路径加载模块
    import sys  # 操作 Python 模块缓存

    reward_fn_config = config.get("custom_reward_function") or {}  # 取出自定义奖励相关配置
    file_path = reward_fn_config.get("path")  # 目标 Python 文件路径
    if not file_path:
        return None  # 未提供路径则表示不使用自定义奖励

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")  # 路径不存在直接报错

    spec = importlib.util.spec_from_file_location("custom_module", file_path)  # 构建动态模块加载规格
    module = importlib.util.module_from_spec(spec)  # 基于规格创建模块对象
    try:
        sys.modules["custom_module"] = module  # 注册到模块缓存，便于相互引用
        spec.loader.exec_module(module)  # 执行模块代码
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}") from e  # 加载失败时抛出异常

    function_name = reward_fn_config.get("name")  # 目标函数名
    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")  # 找不到指定函数时报错

    print(f"using customized reward function '{function_name}' from '{file_path}'")  # 打印使用信息
    raw_fn = getattr(module, function_name)  # 取出原始函数

    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))  # 取出附加参数（例如权重系数）

    def wrapped_fn(*args, **kwargs):  # 外包一层以注入附加参数
        return raw_fn(*args, **kwargs, **reward_kwargs)

    return wrapped_fn  # 返回包装后的函数供训练流程调用


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)  # 指定 Hydra 配置入口
def main(config):  # Hydra 会将解析后的配置对象传入此函数
    run_ppo(config)  # 将控制权交给核心训练逻辑


def run_ppo(config) -> None:  # PPO 训练主流程
    # TODO(linjunrong.ocss884): this ENV is left for resolving SGLang conflict with ray devices
    # isolation, will solve in the future
    # 以上 TODO 保留：用于解决与 SGLang 的设备隔离冲突
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")  # 保留当前 CUDA_VISIBLE_DEVICES 设置
    if not ray.is_initialized():  # 若 Ray 尚未初始化则在本进程内启动
        # this is for local ray cluster
        # 本地模式下初始化 Ray 集群
        ray.init(  # 设置 Ray 本地集群的运行时环境
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN"}},  # 控制关键环境变量
            num_cpus=config.ray_init.num_cpus,  # 根据配置分配 CPU 资源
        )

    runner = TaskRunner.remote()  # 创建远程任务执行器
    ray.get(runner.run.remote(config))  # 在 Ray actor 中执行训练任务并同步等待完成


@ray.remote(num_cpus=1)  # 请确保该 Actor 不运行在 head 节点（原注释），独占 1 个 CPU
class TaskRunner:  # 封装在 Ray Actor 中执行训练逻辑，避免主进程阻塞
    def run(self, config):  # Ray 会在远程环境中调用此方法
        # 打印解析后的完整配置，便于排查
        from pprint import pprint  # 友好打印工具

        from omegaconf import OmegaConf  # Hydra 配置对象操作库

        from verl.utils.fs import copy_to_local  # 文件系统工具，将远端权重复制到本地

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True 会解析引用，打印最终配置
        OmegaConf.resolve(config)  # 就地解析配置中的懒加载字段

        # 从远端（如 HDFS）下载模型权重到本地
        local_path = copy_to_local(config.actor_rollout_ref.model.path)  # 将模型 checkpoint 同步到本地磁盘

        # 初始化分词器与处理器
        from verl.utils import hf_processor, hf_tokenizer  # HuggingFace 工具函数

        trust_remote_code = config.data.get("trust_remote_code", False)  # 是否允许加载自定义代码
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)  # 初始化分词器
        processor = hf_processor(local_path, use_fast=True)  # 初始化多模态处理器，可能为 None

        # 根据策略选择 worker 类
        if config.actor_rollout_ref.actor.strategy == "fsdp":  # FSDP 策略下的 worker 配置
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy  # 确保 actor 与 critic 策略一致
            from verl.single_controller.ray import RayWorkerGroup  # Ray worker 组封装
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker  # FSDP 对应 worker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker  # 根据是否异步选择 rollout worker
            ray_worker_group_cls = RayWorkerGroup  # 使用默认 Ray worker 组

        elif config.actor_rollout_ref.actor.strategy == "megatron":  # Megatron 策略下的 worker 配置
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy  # 仍需保证一致
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup  # Megatron 专用 worker 组
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker  # Megatron worker 实现

            actor_rollout_cls = ActorRolloutRefWorker  # Megatron 目前仅支持同步 rollout
            ray_worker_group_cls = NVMegatronRayWorkerGroup  # 指定 Megatron worker 组

        else:
            raise NotImplementedError  # 其它策略暂未实现

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role  # 资源池管理与角色枚举

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),  # 将 actor rollout 角色绑定到 Ray 远程类
            Role.Critic: ray.remote(CriticWorker),  # 将 critic 角色绑定到 Ray 远程类
        }

        global_pool_id = "global_pool"  # 定义全局资源池名称
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,  # 每个节点提供的 GPU 数
        }
        mapping = {
            Role.ActorRollout: global_pool_id,  # 指定角色使用的资源池
            Role.Critic: global_pool_id,
        }

        # 奖励可能来自多种渠道：
        # - 规则式 RM：直接给出分数
        # - 模型式 RM：调用奖励模型打分
        # - 代码类任务：有测试用例时需调用沙箱
        # - 最终按标签组合不同奖励
        if config.reward_model.enable:  # 若启用奖励模型，则额外分配 worker
            if config.reward_model.strategy == "fsdp":
                from verl.workers.fsdp_workers import RewardModelWorker  # FSDP 奖励模型
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker  # Megatron 奖励模型
            else:
                raise NotImplementedError  # 其它策略暂不支持
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)  # 注册奖励模型角色
            mapping[Role.RewardModel] = global_pool_id  # 使用同一资源池

        # use reference model
        # 是否加载参考模型用于 KL 约束
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:  # 若需 KL 约束则加载参考模型
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)  # 复用 Actor worker 作为参考策略
            mapping[Role.RefPolicy] = global_pool_id

        reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))  # 加载训练阶段奖励计算器
        # 验证阶段与训练阶段保持一致的奖励配置，避免行为差异
        val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {}))  # 加载验证阶段奖励计算器
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)  # 初始化资源池调度器

        trainer = RayPPOTrainer(
            config=config,  # 完整训练配置
            tokenizer=tokenizer,  # 已加载的分词器
            processor=processor,  # 多模态处理器
            role_worker_mapping=role_worker_mapping,  # 角色到 worker 的映射
            resource_pool_manager=resource_pool_manager,  # 资源池管理器
            ray_worker_group_cls=ray_worker_group_cls,  # worker 组实现
            reward_fn=reward_fn,  # 训练奖励计算器
            val_reward_fn=val_reward_fn,  # 验证奖励计算器
        )
        trainer.init_workers()  # 启动各类远程 worker
        trainer.fit()  # 进入主训练循环


if __name__ == "__main__":  # 允许脚本直接作为主程序运行
    main()  # 交给 Hydra 入口处理配置与执行
