#!/usr/bin/env python3
"""
从32k训练数据中采样1k条数据，创建新的训练文件
"""
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description='从大数据集中采样1k条数据')
    parser.add_argument('--input', type=str, 
                       default='/home/admin123/dl/MemAgent/taskutils/memory_data/hotpotqa/hotpotqa_train_32k.parquet',
                       help='输入数据文件路径')
    parser.add_argument('--output', type=str,
                       default='/home/admin123/dl/MemAgent/taskutils/memory_data/hotpotqa/hotpotqa_train_1k.parquet',
                       help='输出数据文件路径')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='采样数量')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    print(f"正在读取数据文件: {args.input}")
    df = pd.read_parquet(args.input)
    print(f"原始数据条数: {len(df)}")
    
    # 采样1k条数据
    if len(df) >= args.num_samples:
        sampled_df = df.sample(n=args.num_samples, random_state=args.seed)
    else:
        print(f"警告: 数据量({len(df)})少于请求的采样数({args.num_samples})，使用全部数据")
        sampled_df = df
    
    print(f"采样后数据条数: {len(sampled_df)}")
    
    # 保存为parquet文件
    sampled_df.to_parquet(args.output, index=False)
    print(f"已保存到: {args.output}")

if __name__ == '__main__':
    main()

