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

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn
from transformers.cache_utils import DynamicCache

from verl import DataProto
from recurrent.utils import enforce_left_padding, create_position_ids
from verl.utils.torch_functional import logprobs_from_logits

from ..base import BaseRollout

__all__ = ["NaiveKVRollout"]


class NaiveKVRollout(BaseRollout):
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
        if pad_to is not None:
            prompts.meta_info["pad_to"] = pad_to
        if generation_kwargs:
            merged_kwargs = dict(prompts.meta_info.get("generation_kwargs", {}))
            merged_kwargs.update(generation_kwargs)
            prompts.meta_info["generation_kwargs"] = merged_kwargs
        stage = prompts.meta_info.get("stage", "decode")
        if stage == "prefill":
            return self._prefill(prompts)
        return self._decode(prompts)

    def _resolve_dtype(self, dtype_str: Optional[str]) -> Optional[torch.dtype]:
        if dtype_str is None:
            return None
        if hasattr(torch, dtype_str):
            return getattr(torch, dtype_str)
        raise ValueError(f"Unsupported dtype string for KV cache: {dtype_str}")

    def _split_kv_by_batch(
        self,
        past_key_values: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        dtype: Optional[torch.dtype],
    ) -> List[List[Tuple[torch.Tensor, torch.Tensor]]]:
        if past_key_values is None:
            raise ValueError("past_key_values is required for splitting.")

        batch_size = past_key_values[0][0].size(0)
        result: List[List[Tuple[torch.Tensor, torch.Tensor]]] = []
        cpu_device = torch.device("cpu")
        for batch_idx in range(batch_size):
            per_sample: List[Tuple[torch.Tensor, torch.Tensor]] = []
            for key, value in past_key_values:
                key_slice = key[batch_idx:batch_idx + 1].contiguous()
                value_slice = value[batch_idx:batch_idx + 1].contiguous()
                target_dtype = dtype or key_slice.dtype
                key_slice = key_slice.to(device=cpu_device, dtype=target_dtype)
                value_slice = value_slice.to(device=cpu_device, dtype=target_dtype)
                per_sample.append((key_slice, value_slice))
            result.append(per_sample)
        return result

    def _stack_kv(
        self,
        kv_list: Optional[List[List[Tuple[torch.Tensor, torch.Tensor]]]],
        device: torch.device,
        dtype: Optional[torch.dtype],
    ) -> Optional[DynamicCache]:
        if not kv_list:
            return None
        if any(sample_cache is None for sample_cache in kv_list):
            return None
        num_layers = len(kv_list[0])
        if num_layers == 0:
            return None
        if any(len(sample_cache) != num_layers for sample_cache in kv_list):
            raise ValueError("Inconsistent layer counts detected across KV caches.")

        stacked: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx in range(num_layers):
            keys = []
            values = []
            for sample_cache in kv_list:
                key, value = sample_cache[layer_idx]
                keys.append(key.to(device=device, dtype=dtype or key.dtype))
                values.append(value.to(device=device, dtype=dtype or value.dtype))
            stacked.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))
        legacy_cache: Tuple[Tuple[torch.Tensor, torch.Tensor], ...] = tuple(stacked)
        return DynamicCache.from_legacy_cache(legacy_cache)

    def _prepend_attention(
        self, attention_mask: torch.Tensor, kv_seq_lens: Sequence[int]
    ) -> torch.Tensor:
        expanded_masks: List[torch.Tensor] = []
        max_total = 0
        for mask, length in zip(attention_mask, kv_seq_lens):
            prefix = torch.ones(length, dtype=mask.dtype, device=mask.device)
            expanded = torch.cat((prefix, mask), dim=0)
            expanded_masks.append(expanded)
            max_total = max(max_total, expanded.size(0))

        padded = [
            F.pad(expanded, (max_total - expanded.size(0), 0))
            for expanded in expanded_masks
        ]
        return torch.stack(padded, dim=0)

    def _prefill(self, prompts: DataProto) -> DataProto:
        input_ids = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        self.module.eval()
        output = self.module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        self.module.train()

        if not hasattr(output, "past_key_values") or output.past_key_values is None:
            raise ValueError("Model did not return past_key_values during prefill.")

        dtype = self._resolve_dtype(prompts.meta_info.get("kv_cache_dtype"))
        per_sample_kv = self._split_kv_by_batch(output.past_key_values, dtype=dtype)
        kv_seq_lens = [
            sample_cache[0][0].size(-2) if sample_cache else 0 for sample_cache in per_sample_kv
        ]

        empty_batch = TensorDict(
            {
                "_dummy": torch.zeros(
                    (input_ids.size(0), 0),
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
            },
            batch_size=input_ids.size(0),
        )
        meta_info = {
            "past_key_values": per_sample_kv,
            "kv_seq_lens": kv_seq_lens,
        }
        return DataProto(batch=empty_batch, meta_info=meta_info)

    def _decode(self, prompts: DataProto) -> DataProto:
        idx = prompts.batch["input_ids"]
        prompt_attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        pad_token_id = prompts.meta_info.get("pad_token_id")
        if pad_token_id is None:
            raise ValueError("pad_token_id must be provided in prompts.meta_info for KV rollout.")
        idx, prompt_attention_mask, fixed = enforce_left_padding(idx, prompt_attention_mask, pad_token_id)
        if fixed:
            position_ids = create_position_ids(prompt_attention_mask)

        eos_token_id = prompts.meta_info["eos_token_id"]
        generation_kwargs = prompts.meta_info.get("generation_kwargs", {})
        response_length = generation_kwargs.get("max_tokens", self.config.response_length)

        temperature = prompts.meta_info.get("temperature", self.config.temperature)
        if temperature == 0:
            temperature = 1.0
        do_sample = prompts.meta_info.get("do_sample", self.config.do_sample)

        reuse_prefill = prompts.meta_info.get("reuse_prefill", True)
        past_key_values_list = prompts.meta_info.get("past_key_values") if reuse_prefill else None
        kv_dtype = self._resolve_dtype(prompts.meta_info.get("kv_cache_dtype"))
        past_key_values = self._stack_kv(
            past_key_values_list,
            device=idx.device,
            dtype=kv_dtype,
        )

        kv_seq_lens = prompts.meta_info.get("kv_seq_lens") if reuse_prefill else None
        if past_key_values is not None and kv_seq_lens:
            if len(kv_seq_lens) != idx.size(0):
                raise ValueError("kv_seq_lens size does not match batch size.")
            kv_offset = torch.tensor(
                kv_seq_lens, device=idx.device, dtype=position_ids.dtype
            ).unsqueeze(1)
            # runtime mask: prepend ones for cached tokens
            runtime_attention_mask = self._prepend_attention(prompt_attention_mask, kv_seq_lens)
            # training mask (for DataProto) should exclude cached prefix
            attention_mask = prompt_attention_mask
            # position ids for current prompt tokens need to start after KV prefix
            position_ids = create_position_ids(prompt_attention_mask) + kv_offset
        else:
            # If not using KV cache, ensure position_ids matches input_ids length
            if position_ids.size(1) != idx.size(1):
                position_ids = position_ids[:, -idx.size(1):]
            runtime_attention_mask = prompt_attention_mask
            attention_mask = prompt_attention_mask

        top_k = generation_kwargs.get("top_k", self.config.top_k)
        if top_k is not None and top_k <= 0:
            top_k = None

        batch_size = idx.size(0)
        prompt_length = idx.size(1)

        self.module.eval()

        prev_attention_mask = torch.ones(
            size=(batch_size, 1),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

        logits_lst = []
        # Encode the whole prompt once (with KV cache prefix if available)
        output = self.module(
            input_ids=idx,
            attention_mask=runtime_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = output.past_key_values
        logits = output.logits[:, -1, :] / temperature

        if top_k is not None:
            k = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, k)
            logits[logits < v[:, [-1]]] = -float("Inf")

        logits_lst.append(logits)
        probs = F.softmax(logits, dim=-1)
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(probs, dim=-1, keepdim=True)

        runtime_attention_mask = torch.cat((runtime_attention_mask, prev_attention_mask), dim=-1)
        attention_mask = torch.cat((attention_mask, prev_attention_mask), dim=-1)

        for token_id in eos_token_id:
            prev_attention_mask = torch.logical_and(idx_next != token_id, prev_attention_mask.bool())
        prev_attention_mask = prev_attention_mask.to(attention_mask.dtype)

        position_ids = torch.cat((position_ids, position_ids[:, -1:] + 1), dim=-1)

        idx = torch.cat((idx, idx_next), dim=1)

        for _ in range(1, response_length):
            output = self.module(
                input_ids=idx_next,
                attention_mask=runtime_attention_mask,
                position_ids=position_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = output.past_key_values
            logits = output.logits[:, -1, :] / temperature

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)
                logits[logits < v[:, [-1]]] = -float("Inf")

            logits_lst.append(logits)
            probs = F.softmax(logits, dim=-1)
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(probs, dim=-1, keepdim=True)

            runtime_attention_mask = torch.cat((runtime_attention_mask, prev_attention_mask), dim=-1)
            attention_mask = torch.cat((attention_mask, prev_attention_mask), dim=-1)

            for token_id in eos_token_id:
                prev_attention_mask = torch.logical_and(idx_next != token_id, prev_attention_mask.bool())
            prev_attention_mask = prev_attention_mask.to(attention_mask.dtype)

            position_ids = torch.cat((position_ids, position_ids[:, -1:] + 1), dim=-1)

            idx = torch.cat((idx, idx_next), dim=1)

        logits = torch.stack(logits_lst, dim=1)
        prompts_out = idx[:, :prompt_length]
        response = idx[:, prompt_length:]
        log_probs = logprobs_from_logits(logits=logits, labels=response)

        pad_to = prompts.meta_info.get("pad_to")
        pad_token_id = prompts.meta_info.get("pad_token_id")
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
            attention_mask = torch.cat((attention_mask, pad_mask), dim=1)

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
