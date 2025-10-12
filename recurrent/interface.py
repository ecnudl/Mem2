# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
from abc import ABC, abstractmethod  # 引入抽象基类与抽象方法工具
from dataclasses import dataclass  # 引入 dataclass 装饰器用于快速定义配置对象
from typing import Any, Optional, Type, List, Union, Dict, Tuple  # 导入常用类型注解
from uuid import uuid4  # 生成唯一 ID，用于样本跟踪
import numpy as np  # 数值计算库，主要用于统计操作

import torch  # PyTorch，用于张量操作
from tensordict import TensorDict  # TensorDict 提供张量字典容器
from omegaconf import DictConfig  # Hydra 的配置字典类型
from transformers import PreTrainedTokenizer, ProcessorMixin  # HuggingFace 分词器基类与预处理器

from verl.protocol import DataProto, DataProtoItem  # VERL 框架的数据协议封装
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn  # 强化学习数据集基类与打包函数


@dataclass  # 使用 dataclass 自动生成初始化等方法
class RConfig:
    """
    Configuration for Multi-turn Policy Optimization.
    Just an interface. Add anything you need in a subclass of it.
    """
    pass  # 占位实现，具体配置应在子类中扩展字段

class RDataset(RLHFDataset):  # 继承 RLHF 数据集，提供递归任务的默认实现
    """
    Dataset for Multi-turn Policy Optimization.
    This class can be used directly as a subclass of RLHFDataset for RecurrentRL
    (if you do not need any new features)

    Overwritten Method:
        - __getitem__: get a single sample
        - get_batch_keys: tensor keys and non-tensor keys, should be contained in the batch.
        - get_collate_fn: collate function for dataloader, default to the same as RLHFDataset.
    
    The inherited methods are hdfs/parquet related methods. 
    Make sure to call super().__init__() in your subclass to reuse RLHFDataset's initializer.
    """
    def __init__(
        self,
        recurrent_config: RConfig,
        data_files: Union[str, list[str]],
        tokenizer: PreTrainedTokenizer,
        data_config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):  # 初始化数据集，沿用基类构造逻辑
        super().__init__(data_files=data_files, tokenizer=tokenizer, config=data_config, processor=processor)  # 调用父类初始化加载数据

    def __getitem__(self, item) -> dict:  # 获取单条样本并附加额外信息
        """
        Enforce subclass to override this method by declaring it as an abstract method.
        If you don't want to change its behavior, just return super().__getitem__(item).
        """
        row_dict = super().__getitem__(item)  # 使用基类逻辑读取一条样本
        # used in validation metrics reduce
        row_dict["sample_uuid"] = str(uuid4())  # 为样本附加唯一 ID，方便评估阶段聚合
        return row_dict  # 返回处理后的样本字典


    def get_bactch_keys(self) -> tuple[list[str], list[str]]:  # 返回批处理需要的键列表
        return ["input_ids", "attention_mask", "position_ids"], []  # 默认返回需要打包的张量键与非张量键

    @staticmethod
    def get_collate_fn():  # 提供 DataLoader collate 函数
        return collate_fn  # 使用 RLHFDataset 提供的默认打包函数

from .async_utils import ChatCompletionProxy  # 引入异步聊天代理，用于与 LLM 服务交互

class AsyncOutput(ABC):  # 定义异步 agent rollout 的返回结果封装
    def __init__(self, 
                 conversations: List[List[Dict[str, str]]],  # 完整对话轮次记录
                 sample_index: int,  # 对应原始样本的索引
                 final_mask: bool,  # 标记该输出是否对应最终回合
                 timing_raw: dict,  # 收集的时间消耗指标
                 metrics: dict = None):  # 可选的额外评估指标；构造函数初始化结果结构
        self.conversations = conversations  # 保存生成的对话内容
        self.sample_index = sample_index  # 记录样本索引
        self.final_mask = final_mask  # 记录是否为结束回合
        self.timing_raw = timing_raw  # 存储原始计时信息
        if metrics is None:
            metrics = {}  # 若未提供指标则使用空字典
        self.metrics = metrics  # 挂载指标字典
        if "workflow/num_conv" not in metrics:
            metrics["workflow/num_conv"] = len(conversations)  # 默认记录生成的轮次数量
    
class AsyncRAgent(ABC):  # 异步递归 Agent 接口定义
    """
    An async recurrent agent interface.

    1. Any const variable that can be created in advance? (__init__)
    2. How to start a new generation? (start)
    3. How to prompt LLM / How to process generated response / When to stop (rollout)
    > note that you should focus on a SINGLE sample instead of a group or a batch.
    """
    def __init__(self, proxy: ChatCompletionProxy, tokenizer:PreTrainedTokenizer, config: RConfig, rollout_config: DictConfig):
        self.proxy = proxy  # 保存异步调度器代理
        self.tokenizer = tokenizer  # 记录分词器用于格式化提示
        self.config = config  # 保存递归配置
        self.rollout_config = rollout_config  # 保存推理相关配置
        self.timing_raw = {}  # 初始化计时字典

    # If you need to initialize/clean up some resource, override this two methods.
    def start(self, gen_batch: DataProto, timing_raw: dict):  # 启动处理单个批次的准备工作
        pass  # 子类实现启动逻辑，例如状态初始化
    def end(self):  # 收尾逻辑，释放资源
        pass  # 子类实现结束逻辑，例如资源释放
        

    @abstractmethod
    async def rollout(self, gen_item: DataProtoItem) -> AsyncOutput:  # 定义单样本的异步生成流程
        """
        Rollout a single sample, returns conversations/sample_index/final_mask + timing/metrics...
        """
        pass
    
    def sampling_params(self, meta_info):  # 根据元信息调整采样参数
        """
        Adapted from works/rollout/vllm_spmd_rollout, returns topp/temperature/n for generation
        Notice that you should specify max_completion_tokens manually, since it can be different for different agents
        Also notice that top_k is not supported in async mode
        """
        kwargs = dict(
                n=1,  # 默认生成一次
                temperature=self.rollout_config.temperature,  # 使用 rollout 配置的温度
                top_p=self.rollout_config.top_p,  # 使用 rollout 配置的核采样阈值
            )
        do_sample = meta_info.get("do_sample", True)  # 读取是否进行采样
        is_validate = meta_info.get("validate", False)  # 读取是否处于验证模式
        if not do_sample:
                # logger.info(f"original {kwargs=}, updating becase do_sample is False")
            kwargs.update({
                    'best_of': 1,  # 贪心时只保留单条最优
                    'top_p': 1.0,  # 关闭 top-p 限制
                    'min_p': 0.0,  # 不考虑最小概率阈值
                    'temperature': 0,  # 设置温度为 0 进行贪心采样
                    'n': 1  # 贪心仅生成一个结果
                })
        elif is_validate:
                # logger.info(f"original {kwargs=}, updating because is_validate is True")
                # TODO: try **
            kwargs.update({
                    'top_p': self.rollout_config.val_kwargs.top_p,  # 验证阶段使用单独的 top-p
                    'temperature': self.rollout_config.val_kwargs.temperature,  # 验证阶段的温度
                    'n': 1,  # 验证时每个样本仅生成一次
                })
            
        return kwargs  # 返回调整后的采样参数


    def reduce_timings(self, timing_raws: list[dict]) -> dict:
        """
        Reduce timing_raw of multiple agents.
        Make sure to follow the naming convention of timing_raw: "async" should be contained in the key,
        if and only if the timed code is an `await` statement.
        """
        reduced = {}  # 聚合后的计时结果
        for k in timing_raws[0]:
            if "async" in k:
                # async method can be executed parallelly
                reduced[k] = sum([timing_raw[k] for timing_raw in timing_raws]) / len(timing_raws)  # 异步部分取平均
            else:
                # sync method is executed sequentially
                reduced[k] = sum([timing_raw[k] for timing_raw in timing_raws])
        return reduced  # 返回合并后的计时信息

    def reduce_metrics(self, metrics: list[dict]) -> dict:
        reduced = {}  # 存放归并后的指标
        for k in metrics[0]:
            reduced[k + "_mean"] = np.mean([m[k] for m in metrics])  # 计算平均值
            reduced[k + "_min"] = np.min([m[k] for m in metrics])  # 计算最小值
            reduced[k + "_max"] = np.max([m[k] for m in metrics])  # 计算最大值
        return reduced  # 返回汇总后的指标字典

class RAgent(ABC):  # 同步递归 Agent 接口定义
    """
    A recurrent agent interface, you should focus on:

    1. Any const variable that can be created in advance? (__init__)
    2. How to start a new generation? (start)
    3. How to prompt LLM? (action)
    4. How to process generated response? (update)
    5. When to stop? (done)
    6. Any resource cleanup? (end)

    All methods are marked as abstract, they WILL NOT be called by default and are just a hint
    about how it should be implemented.
    """
    @abstractmethod
    def __init__(self, tokenizer:PreTrainedTokenizer, config: RConfig):  # 初始化同步 agent 所需资源
        pass  # 子类负责保存分词器与配置等对象
    @abstractmethod
    def start(self, gen_batch: DataProto, timing_raw: dict):  # 处理批次前的准备工作
        """
        Called once at the beginning of generation loop.
        Initialize agent state, store gen_batch and timing_raw.
        """
        self.gen_batch = gen_batch  # 缓存当前批次数据
        self.timing_raw = timing_raw  # 保存计时字典供后续累加
        self.step = 0  # 初始化轮次计数
        self.final_mask_list = [] # only the final turn will be verified, used for reward compute
        self.sample_index_list = [] # map each turn to the sample id in the original batch
        pass  # 子类可扩展额外状态
    @abstractmethod
    def action(self) -> tuple[list[torch.Tensor], dict]:  # 构造下一轮输入与元信息
        """
        Called once for each rollout step.
        Return (input_ids(list[IntTensor]), meta_info).
        Remember to add sample_index to internal state.
        If the agent can decide if the sample is the final turn, also remember to add final_mask,
        else, you can decide in `update`.

        e.g. MemoryAgent will terminate the generation loop after all context is consumed, so it can
        compute a final_mask here
        """
        sample_index = torch.arange(len(self.gen_batch), dtype=torch.long)  # 默认按批次顺序生成索引
        self.sample_index_list.append(sample_index)  # 记录本轮涉及的样本索引
        self.final_mask_list.append(torch.full(sample_index.shape, False, dtype=torch.bool))  # 默认标记为非最终轮
        pass  # 子类需要返回构造好的输入与元信息
    @abstractmethod
    def update(self, gen_output: DataProto) -> DataProto:  # 利用模型输出更新 agent 状态
        """
        Called once after rollout, agent can execute tool calling or other custom action, and update agent state.
        
        e.g. CodeAgnet will terminate the generation loop if there is no code within ```python```.
        """
        pass  # 子类负责解析模型输出更新内部状态
    @abstractmethod
    def done(self):  # 判断生成循环是否结束
        """
        Whether the generation loop should stop.
        """
        return False  # 默认不断开循环，需子类重写结束条件
    @abstractmethod
    def end(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:  # 清理状态并返回最终索引
        """
        Called once after done() returns True.
        `del` the previouly saved data here, `gen_batch` for example.
        Can save some cpu memory(this batch will not be deleted until the next iteration).

        Returns final_mask(bool) and sample_index(long)
        """
        del self.gen_batch  # 释放批次引用
        del self.timing_raw  # 清理计时字典引用
        self.step = 0  # 重置循环计数
        sample_index = torch.cat(self.sample_index_list)  # 拼接所有回合的样本索引
        final_mask = torch.cat(self.final_mask_list)  # 拼接所有回合的终止标记
        del self.final_mask_list  # 释放缓存
        del self.sample_index_list  # 释放缓存
        return final_mask, sample_index  # 返回最终标记与索引，供上层裁剪

@dataclass  # 使用 dataclass 简化注册对象的存储结构
class RRegister:
    """Register your custom recurrent implementation with this class. The register object will be used to create these classes.
    """
    config_cls: Type[RConfig]  # 指向配置类
    dataset_cls: Type[RDataset]  # 指向数据集类
    agent_cls: Type[RAgent]  # 指向 Agent 实现

    @classmethod
    def from_filename(cls, file_path: str, obj_name: str) -> 'RRegister':
        import importlib.util  # 动态导入工具
        import os  # 文件系统操作库
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Recurrent implementation file '{file_path}' not found.")  # 校验文件存在

        spec = importlib.util.spec_from_file_location("custom_module", file_path)  # 构建模块加载规格
        if not spec:
            raise FileNotFoundError(f"Failed to create model spec for '{file_path}'.")  # 规格创建失败时报错
        module = importlib.util.module_from_spec(spec)  # 基于规格创建模块对象
        try:
            spec.loader.exec_module(module)  # 执行模块以加载内容
        except Exception as e:
            raise RuntimeError(f"Error loading module from '{file_path}': {e}")  # 捕获加载异常

        if not hasattr(module, obj_name):
            raise AttributeError(f"Register object '{obj_name}' not found in '{file_path}'.")  # 未找到注册对象时报错

        obj = getattr(module, obj_name)  # 取出指定名称的对象
        if not isinstance(obj, cls):
            raise TypeError(f"Object '{obj_name}' in '{file_path}' is not an instance of {cls}.")  # 确保类型匹配
        print(f"[RECURRENT] recurrent enabled, using register '{obj_name}' from '{file_path}'.")  # 打印加载信息
        return obj  # 返回注册对象供外部使用
