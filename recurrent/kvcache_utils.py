from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch

KVTuple = Tuple[torch.Tensor, torch.Tensor]
KVLike = Union["KVCacheItem", KVTuple]
LayerwiseKV = Optional[Sequence[KVLike]]


@dataclass(frozen=True)
class KVCacheItem:
    """Container for a single layer's KV tensors."""

    key: torch.Tensor
    value: torch.Tensor

    @property
    def seq_len(self) -> int:
        return self.key.size(-2)


def _kv_like_to_item(kv: KVLike) -> KVCacheItem:
    if isinstance(kv, KVCacheItem):
        return kv
    if isinstance(kv, (list, tuple)) and len(kv) == 2:
        key, value = kv
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise TypeError("KV tuple must contain tensors.")
        return KVCacheItem(key=key, value=value)
    raise TypeError(f"Unsupported KV cache item type: {type(kv)}")


def _normalize_kv(
    kv: Optional[Union[KVLike, Sequence[KVLike]]],
) -> Optional[Tuple[KVCacheItem, ...]]:
    if kv is None:
        return None

    if isinstance(kv, (list, tuple)):
        items: List[KVCacheItem] = []
        for layer in kv:
            items.append(_kv_like_to_item(layer))
        return tuple(items)

    return (_kv_like_to_item(kv),)


def _pack_kv(
    items: Optional[Tuple[KVCacheItem, ...]],
    template: Optional[Sequence[KVLike]] = None,
) -> Optional[Union[Tuple[KVTuple, ...], List[KVTuple]]]:
    if items is None:
        return None

    packed: List[KVTuple] = [(item.key, item.value) for item in items]

    if template is None:
        return tuple(packed)

    if isinstance(template, list):
        return packed
    return tuple(packed)


def kv_seq_len(past_kv: LayerwiseKV) -> int:
    normalized = _normalize_kv(past_kv)
    if not normalized:
        return 0

    return normalized[0].seq_len


def extend_attention_mask(
    attention_mask: Optional[torch.Tensor], added_len: int
) -> Optional[torch.Tensor]:
    if attention_mask is None or added_len <= 0:
        return attention_mask

    prefix_shape = (*attention_mask.shape[:-1], added_len)
    prefix = torch.ones(
        prefix_shape, dtype=attention_mask.dtype, device=attention_mask.device
    )

    return torch.cat((prefix, attention_mask), dim=-1)


def extend_position_ids(
    position_ids: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]], added_len: int
) -> Optional[Union[torch.Tensor, Sequence[torch.Tensor]]]:
    if position_ids is None or added_len <= 0:
        return position_ids

    if isinstance(position_ids, torch.Tensor):
        return position_ids + added_len

    if isinstance(position_ids, (list, tuple)):
        extended = [pid + added_len for pid in position_ids]
        return type(position_ids)(extended)

    raise TypeError("Unsupported position_ids type.")


def concat_past_kv(
    external: LayerwiseKV,
    current: LayerwiseKV,
) -> Optional[Union[Tuple[KVTuple, ...], List[KVTuple]]]:
    ext_norm = _normalize_kv(external)
    cur_norm = _normalize_kv(current)

    if ext_norm is None:
        return _pack_kv(cur_norm, current)
    if cur_norm is None:
        return _pack_kv(ext_norm, external)

    if len(ext_norm) != len(cur_norm):
        raise ValueError(
            "Mismatch in number of layers between external and current KV caches."
        )

    merged: List[KVCacheItem] = []
    for ext_item, cur_item in zip(ext_norm, cur_norm):
        if ext_item.key.shape[:-2] != cur_item.key.shape[:-2]:
            raise ValueError("KV cache batch/head dimensions must match before concat.")

        key = torch.cat((ext_item.key, cur_item.key), dim=-2)
        value = torch.cat((ext_item.value, cur_item.value), dim=-2)
        merged.append(KVCacheItem(key=key, value=value))

    return _pack_kv(tuple(merged), current or external)
