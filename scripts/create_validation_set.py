#!/usr/bin/env python3
"""
Create a validation set from HotpotQA dev data
that matches the training data distribution.

Usage:
    python create_validation_set.py --num_samples 100 --max_length 30000
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def create_validation_set(
    input_path: str,
    output_path: str,
    num_samples: int = 100,
    max_length: int = 30000,
    min_length: int = 20000,
    seed: int = 42
):
    """
    Create a validation set by sampling from dev data.

    Args:
        input_path: Path to source dev parquet file
        output_path: Path to save output validation set
        num_samples: Number of samples to include
        max_length: Maximum context length (in tokens, estimated as chars/4)
        min_length: Minimum context length
        seed: Random seed for reproducibility
    """
    print("="*80)
    print("Creating Validation Set")
    print("="*80)

    # Load data
    print(f"\n1. Loading data from: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"   Total samples: {len(df)}")

    # Calculate context lengths
    if 'context_tokens' not in df.columns:
        df['context_tokens'] = df['context'].str.len() / 4

    print(f"\n2. Original context length distribution:")
    print(f"   Mean: {df['context_tokens'].mean():.0f} tokens")
    print(f"   Median: {df['context_tokens'].median():.0f} tokens")
    print(f"   Min: {df['context_tokens'].min():.0f} tokens")
    print(f"   Max: {df['context_tokens'].max():.0f} tokens")

    # Filter by length
    print(f"\n3. Filtering samples with length {min_length}-{max_length} tokens")
    df_filtered = df[
        (df['context_tokens'] >= min_length) &
        (df['context_tokens'] <= max_length)
    ].copy()
    print(f"   Samples after filtering: {len(df_filtered)}")

    if len(df_filtered) == 0:
        print(f"\n❌ ERROR: No samples found in range [{min_length}, {max_length}]")
        print(f"   Try adjusting --min_length and --max_length parameters")
        return

    if len(df_filtered) < num_samples:
        print(f"\n⚠️  WARNING: Only {len(df_filtered)} samples available, but {num_samples} requested")
        print(f"   Will use all {len(df_filtered)} samples")
        num_samples = len(df_filtered)

    # Sample
    print(f"\n4. Randomly sampling {num_samples} examples (seed={seed})")
    np.random.seed(seed)
    df_sampled = df_filtered.sample(n=num_samples, random_state=seed)

    # Sort by length for better batching
    df_sampled = df_sampled.sort_values('context_tokens')

    print(f"\n5. Final validation set statistics:")
    print(f"   Samples: {len(df_sampled)}")
    print(f"   Context length:")
    print(f"     Mean: {df_sampled['context_tokens'].mean():.0f} tokens")
    print(f"     Median: {df_sampled['context_tokens'].median():.0f} tokens")
    print(f"     Min: {df_sampled['context_tokens'].min():.0f} tokens")
    print(f"     Max: {df_sampled['context_tokens'].max():.0f} tokens")
    print(f"     Std: {df_sampled['context_tokens'].std():.0f} tokens")

    # Save
    print(f"\n6. Saving to: {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_sampled.to_parquet(output_path, index=False)

    file_size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"   File size: {file_size_mb:.1f} MB")

    print("\n" + "="*80)
    print("✅ Validation set created successfully!")
    print("="*80)
    print(f"\nUsage in training script:")
    print(f'   VAL_PATH="{output_path}"')

    # Estimate validation time
    val_rollout_n = 4
    total_generations = len(df_sampled) * val_rollout_n
    time_estimate = total_generations * 2 / 60  # 2 sec per generation
    print(f"\nEstimated validation time:")
    print(f"   {len(df_sampled)} samples × {val_rollout_n} rollout = {total_generations} generations")
    print(f"   ~{time_estimate:.1f} minutes per validation run")


def main():
    parser = argparse.ArgumentParser(
        description="Create validation set from HotpotQA dev data"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='/home/admin123/dl/MemAgent/taskutils/memory_data/hotpotqa/hotpotqa_dev.parquet',
        help='Path to source dev parquet file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/home/admin123/dl/MemAgent/taskutils/memory_data/hotpotqa/hotpotqa_dev_100_filtered.parquet',
        help='Path to save output validation set'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=100,
        help='Number of samples to include (default: 100)'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=30000,
        help='Maximum context length in tokens (default: 30000)'
    )
    parser.add_argument(
        '--min_length',
        type=int,
        default=20000,
        help='Minimum context length in tokens (default: 20000)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    create_validation_set(
        input_path=args.input,
        output_path=args.output,
        num_samples=args.num_samples,
        max_length=args.max_length,
        min_length=args.min_length,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
