from __future__ import annotations

from recurrent.generation_manager_kv import LLMGenerationManager as KVLLMGenerationManager

from .ray_trainer import RayPPOTrainer


class RayPPOTrainerKV(RayPPOTrainer):
    """RayPPOTrainer 的 KVcache 版本，重用大部分逻辑，仅改写生成管理器。"""

    def init_workers(self):
        super().init_workers()

        if not self.config.recurrent.enable:
            return
        if self.async_rollout_mode:
            raise NotImplementedError("KVcache 记忆暂不支持 async rollout 模式。")

        self.generation_manager = KVLLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=self.recurrent_config,
            agent_cls=self.recurrent_register.agent_cls,
        )
