#!/usr/bin/env python3
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
LoRA Model Merger for FSDP Checkpoints

This script merges FSDP-sharded LoRA checkpoints into a single HuggingFace model
by first merging FSDP shards, then merging LoRA weights into base weights.

Usage:
    python scripts/lora_merger.py \
        --hf_model_path /path/to/base_model \
        --local_dir /path/to/checkpoint/actor \
        --target_dir /path/to/output \
        --lora_rank 64 \
        --lora_alpha 32
"""

import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

import numpy as np
import torch
from safetensors.torch import load_file
from torch.distributed._tensor import Placement, Shard
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    GenerationConfig,
)

try:
    # for torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

parser = argparse.ArgumentParser()
parser.add_argument("--hf_model_path", type=str, required=True, help="Path to base HuggingFace model")
parser.add_argument("--local_dir", type=str, required=True, help="Path to FSDP checkpoint directory")
parser.add_argument("--target_dir", required=True, type=str, help="Output directory for merged model")
parser.add_argument("--lora_rank", type=int, default=64, help="LoRA rank (r)")
parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling factor")
parser.add_argument("--tie-word-embedding", action="store_true", help="Whether to tie word embedding weights")
args = parser.parse_args()

os.makedirs(args.target_dir, exist_ok=True)


def merge_by_placement(tensors: List[torch.Tensor], placement: Placement):
    """Merge tensors based on their FSDP placement strategy"""
    if placement.is_replicate():
        return tensors[0]
    elif placement.is_partial():
        raise NotImplementedError("Partial placement is not supported yet")
    elif placement.is_shard():
        return torch.cat(tensors, dim=placement.dim).contiguous()
    else:
        raise ValueError(f"Unsupported placement: {placement}")


def patch_model_generation_config(model, hf_model_path):
    """Patch generation config from pretrained model"""
    if model.can_generate():
        try:
            model.generation_config = GenerationConfig.from_pretrained(hf_model_path)
        except OSError:
            print(
                f"Warning: Generation config file not found in {hf_model_path}, "
                "using a generation config created from the model config."
            )
            pass
    return model


def load_and_merge_fsdp_shards() -> Tuple[Dict[str, torch.Tensor], Dict[str, List[Placement]]]:
    """Load FSDP shards and merge them into a single state dict (with LoRA structure intact)"""
    local_dir = args.local_dir

    # Find world size from checkpoint filenames
    rank = 0
    world_size = 0
    for filename in os.listdir(local_dir):
        match = re.match(r"model_world_size_(\d+)_rank_0\.pt", filename)
        if match:
            world_size = match.group(1)
            break
    assert world_size, "No model file with the proper format"

    print(f"Found FSDP checkpoint with world_size={world_size}")

    # Load rank 0 to get sharding info
    state_dict = torch.load(
        os.path.join(local_dir, f"model_world_size_{world_size}_rank_{rank}.pt"),
        map_location="cpu",
        weights_only=False,
    )
    pivot_key = sorted(list(state_dict.keys()))[0]
    weight = state_dict[pivot_key]

    if isinstance(weight, DTensor):
        device_mesh = weight.device_mesh
        mesh = device_mesh.mesh
        mesh_dim_names = device_mesh.mesh_dim_names
    else:
        mesh = np.array([int(world_size)], dtype=np.int64)
        mesh_dim_names = ("fsdp",)

    print(f"Got device mesh {mesh}, mesh_dim_names {mesh_dim_names}")
    assert mesh_dim_names in (("fsdp",), ("ddp", "fsdp")), f"Unsupported mesh_dim_names {mesh_dim_names}"

    if "tp" in mesh_dim_names:
        total_shards = mesh.shape[-1] * mesh.shape[-2]
        mesh_shape = (mesh.shape[-2], mesh.shape[-1])
    else:
        total_shards = mesh.shape[-1]
        mesh_shape = (mesh.shape[-1],)

    print(f"Processing {total_shards} shards with shape {mesh_shape}")

    # Load all shards
    model_state_dict_lst = [state_dict]
    model_state_dict_lst.extend([""] * (total_shards - 1))

    def process_one_shard(rank, model_state_dict_lst):
        model_path = os.path.join(local_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        model_state_dict_lst[rank] = state_dict
        return state_dict

    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count())) as executor:
        for rank in range(1, total_shards):
            executor.submit(process_one_shard, rank, model_state_dict_lst)

    # Merge shards
    merged_state_dict = {}
    param_placements: Dict[str, List[Placement]] = {}
    keys = set(model_state_dict_lst[0].keys())

    for key in keys:
        merged_state_dict[key] = []
        for model_state_dict in model_state_dict_lst:
            tensor = model_state_dict.pop(key)
            if isinstance(tensor, DTensor):
                merged_state_dict[key].append(tensor._local_tensor.bfloat16())
                placements = tuple(tensor.placements)
                if mesh_dim_names[0] in ("dp", "ddp"):
                    placements = placements[1:]
                if key not in param_placements:
                    param_placements[key] = placements
                else:
                    assert param_placements[key] == placements
            else:
                merged_state_dict[key].append(tensor.bfloat16())

    del model_state_dict_lst

    # Merge according to placements
    for key in sorted(merged_state_dict):
        if not isinstance(merged_state_dict[key], list):
            continue
        if key in param_placements:
            placements: Tuple[Shard] = param_placements[key]
            if len(mesh_shape) == 1:
                shards = merged_state_dict[key]
                merged_state_dict[key] = merge_by_placement(shards, placements[0])
            else:
                raise NotImplementedError("FSDP + TP is not supported yet")
        else:
            merged_state_dict[key] = torch.cat(merged_state_dict[key], dim=0)

    print(f"Merged FSDP shards into {len(merged_state_dict)} tensors")
    return merged_state_dict


def merge_lora_weights(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Merge LoRA weights into base weights and remove wrapper structure.

    Input state_dict has keys like:
        base_model.model.lm_head.weight
        base_model.model.model.layers.0.mlp.down_proj.base_layer.weight
        base_model.model.model.layers.0.mlp.down_proj.lora_A.default.weight
        base_model.model.model.layers.0.mlp.down_proj.lora_B.default.weight

    Output state_dict has keys like:
        lm_head.weight
        model.layers.0.mlp.down_proj.weight

    With LoRA merged: weight = base_layer.weight + (lora_B @ lora_A) * scaling
    """
    scaling = args.lora_alpha / args.lora_rank
    print(f"Merging LoRA weights with scaling factor: {scaling} (alpha={args.lora_alpha}, rank={args.lora_rank})")

    # Group keys by base parameter name
    lora_groups = {}
    base_keys = set()

    for key in state_dict.keys():
        # Check if this is a LoRA parameter
        if ".lora_A." in key or ".lora_B." in key:
            # Extract base parameter name: remove .lora_A/B.default suffix
            base_key = key.replace(".lora_A.default", "").replace(".lora_B.default", "")
            if base_key not in lora_groups:
                lora_groups[base_key] = {}

            if ".lora_A." in key:
                lora_groups[base_key]["lora_A"] = key
            else:
                lora_groups[base_key]["lora_B"] = key
        elif ".base_layer." in key:
            # This is a base layer with LoRA adapters
            base_key = key.replace(".base_layer", "")
            if base_key not in lora_groups:
                lora_groups[base_key] = {}
            lora_groups[base_key]["base_layer"] = key
            base_keys.add(base_key)
        elif key.startswith("base_model.model."):
            # This is a parameter without LoRA (e.g., layer norms, embeddings)
            base_keys.add(key)

    # Create merged state dict
    merged_state_dict = {}
    lora_merged_count = 0

    for key in state_dict.keys():
        # Skip lora_A and lora_B keys (they'll be merged)
        if ".lora_A." in key or ".lora_B." in key:
            continue

        # Handle base_layer keys (need to merge LoRA)
        if ".base_layer." in key:
            base_key = key.replace(".base_layer", "")

            if base_key in lora_groups and "lora_A" in lora_groups[base_key] and "lora_B" in lora_groups[base_key]:
                # Merge LoRA weights
                base_weight = state_dict[key]
                lora_A = state_dict[lora_groups[base_key]["lora_A"]]
                lora_B = state_dict[lora_groups[base_key]["lora_B"]]

                # Compute delta: lora_B @ lora_A * scaling
                # LoRA formula: W = W_base + BA * (alpha / r)
                delta = (lora_B @ lora_A) * scaling
                merged_weight = base_weight + delta

                # Remove base_model.model. prefix
                clean_key = base_key.replace("base_model.model.", "")
                merged_state_dict[clean_key] = merged_weight
                lora_merged_count += 1
            else:
                # Base layer without LoRA adapters (shouldn't happen in our case)
                clean_key = base_key.replace("base_model.model.", "")
                merged_state_dict[clean_key] = state_dict[key]
        else:
            # Regular parameter (no LoRA, no base_layer)
            # Remove base_model.model. prefix
            if key.startswith("base_model.model."):
                clean_key = key.replace("base_model.model.", "")
                merged_state_dict[clean_key] = state_dict[key]
            else:
                # Keep as-is (shouldn't happen)
                merged_state_dict[key] = state_dict[key]

    print(f"Merged {lora_merged_count} LoRA adapter pairs into base weights")
    print(f"Final state dict has {len(merged_state_dict)} parameters")

    # Sanity check: print first few keys
    print("\nSample merged keys (first 10):")
    for i, key in enumerate(sorted(merged_state_dict.keys())[:10]):
        print(f"  {key}: {merged_state_dict[key].shape}")

    return merged_state_dict


def save_merged_model(state_dict: Dict[str, torch.Tensor]):
    """Save merged model to target directory"""
    print(f"Saving merged model to {args.target_dir}")

    config = AutoConfig.from_pretrained(args.hf_model_path)

    # Create empty model on meta device
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    model.to_empty(device="cpu")
    model = patch_model_generation_config(model, args.hf_model_path)

    # Save model with merged weights
    model.save_pretrained(args.target_dir, state_dict=state_dict)

    print(f"✓ Successfully saved merged model to {args.target_dir}")


def main():
    print("="*60)
    print("LoRA Model Merger for FSDP Checkpoints")
    print("="*60)
    print(f"Base model: {args.hf_model_path}")
    print(f"Checkpoint: {args.local_dir}")
    print(f"Output: {args.target_dir}")
    print(f"LoRA config: rank={args.lora_rank}, alpha={args.lora_alpha}")
    print("="*60)
    print()

    # Step 1: Load and merge FSDP shards
    print("Step 1: Loading and merging FSDP shards...")
    fsdp_merged_state_dict = load_and_merge_fsdp_shards()
    print(f"✓ FSDP merge complete: {len(fsdp_merged_state_dict)} tensors\n")

    # Step 2: Merge LoRA weights
    print("Step 2: Merging LoRA adapters into base weights...")
    final_state_dict = merge_lora_weights(fsdp_merged_state_dict)
    print(f"✓ LoRA merge complete: {len(final_state_dict)} parameters\n")

    # Step 3: Save merged model
    print("Step 3: Saving merged model...")
    save_merged_model(final_state_dict)
    print(f"✓ Model saved successfully\n")

    print("="*60)
    print("✓ LoRA merge completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
