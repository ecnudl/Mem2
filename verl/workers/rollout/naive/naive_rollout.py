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
In single GPU rollout, the sequences are generated directly by sampling from the model.
The output will contain
1. output_ids
2. attention_masks (left padding)
3. eos_masks
4. log_probs
"""

from typing import Optional

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

from verl import DataProto
from verl.utils.torch_functional import logprobs_from_logits

from ..base import BaseRollout

__all__ = ["NaiveRollout"]


class NaiveRollout(BaseRollout):
    def __init__(self, module: nn.Module, config):
        """A naive rollout. It requires the module to be compatible with huggingface APIs. That is:
        The module should define __call__ to receive input_ids, attention_mask and position_ids.
        It outputs a structure that contains logits field.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
        """
        super().__init__()
        self.config = config
        self.module = module

    @torch.no_grad()
    def generate_sequences(
        self,
        prompts: DataProto,
        pad_to: Optional[int] = None,
        **generation_kwargs,
    ) -> DataProto:
        """Generate sequences"""
        meta_info = prompts.meta_info
        if pad_to is not None:
            prompts.meta_info["pad_to"] = pad_to
        if generation_kwargs:
            merged_kwargs = dict(prompts.meta_info.get("generation_kwargs", {}))
            merged_kwargs.update(generation_kwargs)
            prompts.meta_info["generation_kwargs"] = merged_kwargs

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask = prompts.batch["attention_mask"]  # left-padded attention_mask
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = meta_info["eos_token_id"]

        batch_size = idx.size(0)
        prompt_length = idx.size(1)

        generation_kwargs = meta_info.get("generation_kwargs", {})
        response_steps = generation_kwargs.get("max_tokens", self.config.response_length)
        temperature = meta_info.get("temperature", self.config.temperature)
        if temperature == 0:
            temperature = 1.0
        do_sample = meta_info.get("do_sample", self.config.do_sample)
        top_k = generation_kwargs.get("top_k", self.config.top_k)
        if top_k is not None and top_k <= 0:
            top_k = None

        self.module.eval()

        prev_attention_mask = torch.ones(size=(batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)

        logits_lst = []
        for _ in range(response_steps):
            # if the sequence context is growing too long we must crop it at block_size
            # idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            idx_cond = idx
            # forward the model to get the logits for the index in the sequence
            # we use huggingface APIs here
            output = self.module(input_ids=idx_cond, attention_mask=attention_mask, position_ids=position_ids)
            logits = output.logits
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature  # (bs, vocab_size)
            # optionally crop the logits to only the top k options
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(probs, dim=-1, keepdim=True)

            attention_mask = torch.cat((attention_mask, prev_attention_mask), dim=-1)

            for token_id in eos_token_id:
                prev_attention_mask = torch.logical_and(idx_next != token_id, prev_attention_mask.bool())
            prev_attention_mask.to(attention_mask.dtype)

            position_ids = torch.cat((position_ids, position_ids[:, -1:] + 1), dim=-1)

            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            logits_lst.append(logits)

        logits = torch.stack(logits_lst, dim=1)  # (bs, response_length, vocab_size)
        prompts_out = idx[:, :prompt_length]  # (bs, prompt_length)
        response = idx[:, prompt_length:]  # (bs, response_length)
        log_probs = logprobs_from_logits(logits=logits, labels=response)

        pad_to = meta_info.get("pad_to")
        pad_token_id = meta_info.get("pad_token_id")
        if (
            pad_to is not None
            and pad_token_id is not None
            and response.size(1) < pad_to
        ):
            pad_len = pad_to - response.size(1)
            pad_tokens = torch.full(
                (batch_size, pad_len),
                pad_token_id,
                dtype=response.dtype,
                device=response.device,
            )
            response = torch.cat((response, pad_tokens), dim=1)
            zero_log_probs = torch.zeros(
                (batch_size, pad_len),
                dtype=log_probs.dtype,
                device=log_probs.device,
            )
            log_probs = torch.cat((log_probs, zero_log_probs), dim=1)
            idx = torch.cat((idx, pad_tokens), dim=1)

            pad_mask = torch.zeros(
                (batch_size, pad_len),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat((attention_mask, pad_mask), dim=-1)

            delta_position_id = torch.arange(
                1,
                pad_len + 1,
                device=position_ids.device,
                dtype=position_ids.dtype,
            ).unsqueeze(0).repeat(batch_size, 1)
            position_ids = torch.cat(
                (position_ids, position_ids[:, -1:] + delta_position_id), dim=-1
            )

        batch = TensorDict(
            {
                "prompts": prompts_out,
                "responses": response,
                "sequences": idx,
                "old_log_probs": log_probs,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        if not (idx.size(1) == attention_mask.size(1) == position_ids.size(1)):
            raise ValueError(
                "Sequence length mismatch among input_ids, attention_mask, and position_ids."
            )
        batch["input_ids"] = idx

        self.module.train()

        return DataProto(batch=batch)
