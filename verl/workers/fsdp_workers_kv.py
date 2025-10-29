from __future__ import annotations

from verl.workers.fsdp_workers import (
    ActorRolloutRefWorker as BaseActorRolloutRefWorker,
    AsyncActorRolloutRefWorker as BaseAsyncActorRolloutRefWorker,
    CriticWorker,
    RewardModelWorker,
)


class ActorRolloutRefWorkerKV(BaseActorRolloutRefWorker):
    """FSDP actor/rollout worker with HF KV-cache aware rollout support."""

    def _build_rollout(self, trust_remote_code: bool = False):
        rollout_name = self.config.rollout.name
        if rollout_name == "hf_kv":
            from verl.workers.rollout.naive.naive_rollout_kv import NaiveKVRollout
            from verl.workers.sharding_manager.base import BaseShardingManager

            rollout = NaiveKVRollout(module=self.actor_module_fsdp, config=self.config.rollout)
            rollout_sharding_manager = BaseShardingManager()
            return rollout, rollout_sharding_manager

        return super()._build_rollout(trust_remote_code=trust_remote_code)


class AsyncActorRolloutRefWorkerKV(BaseAsyncActorRolloutRefWorker):
    """Placeholder async worker to keep interface parity; currently hf_kv is sync-only."""

    def _build_rollout(self, trust_remote_code: bool = False):
        if self.config.rollout.name == "hf_kv":
            raise NotImplementedError("hf_kv rollout 目前只支持同步推理模式。")
        return super()._build_rollout(trust_remote_code=trust_remote_code)


__all__ = [
    "ActorRolloutRefWorkerKV",
    "AsyncActorRolloutRefWorkerKV",
    "CriticWorker",
    "RewardModelWorker",
]
