import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from transformers import PreTrainedTokenizer
from typing_extensions import override

from recurrent.impls.memory import MemoryConfig, MemoryDataset
from recurrent.interface import RAgent, RDataset, RRegister
from recurrent.kvcache_utils import concat_past_kv, kv_seq_len, truncate_past_kv
from recurrent.utils import (
    TokenTemplate,
    chat_template,
    create_attention_mask,
    create_position_ids,
    pad_tensor_list_to_length,
)
from verl.protocol import DataProto

logger = logging.getLogger(__file__)
logger.setLevel("INFO")


@dataclass
class KVCacheMemoryConfig(MemoryConfig):
    """
    Extension of MemoryConfig that enables KV cache based memorisation.

    kv_cache_max_length: optional upper bound on cached sequence length. If set to 0/None,
    the cache length will grow with the consumed context.
    kv_cache_dtype: optionally cast cached tensors (e.g. "float16", "bfloat16").
    """

    kv_cache_max_length: Optional[int] = None
    kv_cache_dtype: Optional[str] = None
    reuse_prefill: bool = True
    prompt_as_first_chunk: bool = True


class KVCacheMemoryDataset(MemoryDataset):
    """
    Reuse the existing MemoryDataset behaviour. This subclass exists mainly for clearer
    registration of the KV cache agent.
    """

    pass


class KVCacheMemoryAgent(RAgent):
    """
    Agent that streams document chunks through the policy model using KV cache instead of
    textual memory. Each chunk is fed via a prefill-only forward pass to collect past_key_values.
    The final answer is generated in a single decoding round that reuses the accumulated cache.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, config: KVCacheMemoryConfig):
        self.config = config
        self.tokenizer = tokenizer
        self.chat_template = chat_template(tokenizer)
        final_template = (
            "You are given a problem. Use the context you have just read to answer it.\n\n"
            "<problem>\n{prompt}\n</problem>\n\nAnswer:"
        )
        self.token_final_message_template = TokenTemplate(
            self.chat_template.format(message=final_template), tokenizer
        )
        self.pad_token_id = tokenizer.pad_token_id
        self.pending_tensors = None

    @override
    def start(self, gen_batch: DataProto, timing_raw: dict):
        self.gen_batch = gen_batch
        self.timing_raw = timing_raw
        self.ctx_length = gen_batch.batch["context_length"]
        self.context_ids = gen_batch.batch["context_ids"]
        self.prompt_ids = gen_batch.non_tensor_batch["prompt_ids"]

        self.bsz = len(self.ctx_length)
        self.ctx_offset = torch.zeros(self.bsz, dtype=torch.long)
        self.phase = "prefill"  # prefill -> decode -> done
        self.kv_cache = [None] * self.bsz
        self.last_active_index = torch.zeros(0, dtype=torch.long)
        self.final_mask_list: List[torch.Tensor] = []
        self.sample_index_list: List[torch.Tensor] = []
        self.final_output: Optional[DataProto] = None
        self.pending_tensors = None

    def _context_active_mask(self) -> torch.Tensor:
        return self.ctx_offset < self.ctx_length

    def _prepare_prefill_sequences(self) -> Tuple[List[torch.Tensor], dict]:
        active_mask = self._context_active_mask()
        if active_mask.sum().item() == 0:
            # switch to decode stage
            self.phase = "decode"
            return [], {}

        active_indices = torch.arange(self.bsz, dtype=torch.long)[active_mask]
        self.last_active_index = active_indices
        sequences: List[torch.Tensor] = []

        for idx in active_indices.tolist():
            consumed = self.ctx_offset[idx].item()
            total = self.ctx_length[idx].item()
            next_end = min(consumed + self.config.chunk_size, total)
            chunk = self.context_ids[idx, consumed:next_end].to(torch.long)

            if consumed == 0 and self.config.prompt_as_first_chunk:
                prompt_tensor = torch.tensor(
                    self.prompt_ids[idx], dtype=torch.long
                )
                sequence = torch.cat([prompt_tensor, chunk], dim=0)
            else:
                sequence = chunk

            sequences.append(sequence)
            self.ctx_offset[idx] = next_end

        input_ids = pad_tensor_list_to_length(
            sequences,
            pad_token_id=self.pad_token_id,
            left_pad=True,
        )
        attention_mask = create_attention_mask(input_ids, pad_token_id=self.pad_token_id)
        position_ids = create_position_ids(attention_mask)

        self.pending_tensors = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

        meta_info = {
            "stage": "prefill",
            "input_pad_to": input_ids.size(-1),
            "pad_to": input_ids.size(-1),
            "kv_cache_dtype": self.config.kv_cache_dtype,
            "reuse_prefill": self.config.reuse_prefill,
        }

        return sequences, meta_info

    def _prepare_decode_sequences(self) -> Tuple[List[torch.Tensor], dict]:
        prompts = self.prompt_ids
        messages = [
            self.token_final_message_template.format(prompt=prompt)
            for prompt in prompts
        ]
        sample_index = torch.arange(self.bsz, dtype=torch.long)
        final_mask = torch.ones_like(sample_index, dtype=torch.bool)
        self.sample_index_list.append(sample_index)
        self.final_mask_list.append(final_mask)

        meta_info = {
            "stage": "decode",
            "input_pad_to": max(len(msg) for msg in messages),
            "pad_to": self.config.gen_pad_to,
            "generation_kwargs": {
                "max_tokens": self.config.gen_max_tokens_final_response,
                "n": 1,
            },
            "past_key_values": self.kv_cache,
            "kv_seq_lens": [kv_seq_len(item) for item in self.kv_cache],
            "reuse_prefill": self.config.reuse_prefill,
        }
        self.pending_tensors = None
        return messages, meta_info

    @override
    def action(self) -> Tuple[List[torch.Tensor], dict]:
        if self.phase == "prefill":
            sequences, meta_info = self._prepare_prefill_sequences()
            if sequences:
                self.current_stage = "prefill"
                self.current_meta_info = meta_info
                return sequences, meta_info
            # no sequences produced, move to decode path
            self.phase = "decode"
            return self.action()
            # if no sequences produced, fall through to decode

        if self.phase == "decode":
            messages, meta_info = self._prepare_decode_sequences()
            self.phase = "decode_wait"
            self.current_stage = "decode"
            self.current_meta_info = meta_info
            return messages, meta_info

        raise RuntimeError("KVCacheMemoryAgent.action called in invalid phase.")

    @override
    def update(self, gen_output: DataProto) -> DataProto:
        if getattr(self, "current_stage", None) == "prefill":
            if "past_key_values" not in gen_output.meta_info:
                raise ValueError("Prefill update expects past_key_values in meta_info.")

            past_key_values = gen_output.meta_info["past_key_values"]
            if past_key_values is None:
                raise ValueError("past_key_values is None in prefill update.")

            max_cache_len = self.config.kv_cache_max_length
            for local_idx, sample_idx in enumerate(self.last_active_index.tolist()):
                cache_fragment = past_key_values[local_idx]
                if self.kv_cache[sample_idx] is None:
                    cache = cache_fragment
                else:
                    cache = concat_past_kv(
                        self.kv_cache[sample_idx], cache_fragment
                    )
                cache = truncate_past_kv(cache, max_cache_len)
                self.kv_cache[sample_idx] = cache

            self.current_stage = None
            return gen_output

        if getattr(self, "current_stage", None) == "decode":
            self.final_output = gen_output
            self.phase = "done"
            self.current_stage = None
            return gen_output

        raise RuntimeError("KVCacheMemoryAgent.update called before action.")

    @override
    def done(self) -> bool:
        if self.phase == "prefill":
            # continue streaming until all context consumed
            return False
        if self.phase == "decode":
            return False
        if self.phase == "decode_wait":
            return False
        return self.phase == "done"

    @override
    def end(self):
        sample_index = torch.cat(self.sample_index_list) if self.sample_index_list else torch.arange(self.bsz, dtype=torch.long)
        final_mask = torch.cat(self.final_mask_list) if self.final_mask_list else torch.ones(self.bsz, dtype=torch.bool)
        return final_mask, sample_index


REGISTER = RRegister(
    config_cls=KVCacheMemoryConfig,
    dataset_cls=KVCacheMemoryDataset,
    agent_cls=KVCacheMemoryAgent,
)
