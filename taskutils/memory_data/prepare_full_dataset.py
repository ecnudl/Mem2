#!/usr/bin/env python3
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
"""
生成完整的HotpotQA数据集
- 训练集：32k样本 -> hotpotqa_train_32k.parquet
- 验证集：完整验证集 -> hotpotqa_dev.parquet
"""
import json
import os
import random
import sys
from pathlib import Path
import pandas as pd

# 添加当前目录到路径，以便导入processing模块
sys.path.insert(0, str(Path(__file__).parent))
from processing import read_hotpotqa, generate_dataset

# 设置随机种子
random.seed(42)

def main():
    # 数据目录
    data_dir = Path(__file__).parent
    output_dir = data_dir / "hotpotqa"
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("开始生成完整的HotpotQA数据集")
    print("=" * 60)
    
    # 读取原始数据
    print("\n1. 读取原始JSON数据...")
    train_json_path = data_dir / "hotpotqa_train.json"
    dev_json_path = data_dir / "hotpotqa_dev.json"
    
    if not train_json_path.exists():
        print(f"错误: 找不到训练集文件 {train_json_path}")
        print("请先运行 download_qa_dataset.sh 下载数据")
        return
    
    if not dev_json_path.exists():
        print(f"错误: 找不到验证集文件 {dev_json_path}")
        print("请先运行 download_qa_dataset.sh 下载数据")
        return
    
    QAS_train, DOCS_train = read_hotpotqa(str(train_json_path))
    QAS_dev, DOCS_dev = read_hotpotqa(str(dev_json_path))
    
    print(f"训练集: {len(QAS_train)} 个问题, {len(DOCS_train)} 个文档")
    print(f"验证集: {len(QAS_dev)} 个问题, {len(DOCS_dev)} 个文档")
    
    # 生成训练集 (32k样本)
    print("\n2. 生成训练集 (32k样本)...")
    train_output_path = output_dir / "hotpotqa_train_32k.parquet"
    if train_output_path.exists():
        print(f"训练集已存在: {train_output_path}")
        response = input("是否重新生成? (y/n): ")
        if response.lower() != 'y':
            print("跳过训练集生成")
        else:
            print("生成32k训练集...")
            generate_dataset(
                num_samples=32000,
                save_dir=str(train_output_path).replace('.parquet', ''),
                incremental=200,  # 每个样本使用200个文档
                qas=QAS_train,
                docs=DOCS_train
            )
            print(f"✓ 训练集已保存: {train_output_path}")
    else:
        print("生成32k训练集...")
        generate_dataset(
            num_samples=32000,
            save_dir=str(train_output_path).replace('.parquet', ''),
            incremental=200,  # 每个样本使用200个文档
            qas=QAS_train,
            docs=DOCS_train
        )
        print(f"✓ 训练集已保存: {train_output_path}")
    
    # 生成验证集 (完整验证集)
    print("\n3. 生成验证集 (完整验证集)...")
    dev_output_path = output_dir / "hotpotqa_dev.parquet"
    if dev_output_path.exists():
        print(f"验证集已存在: {dev_output_path}")
        response = input("是否重新生成? (y/n): ")
        if response.lower() != 'y':
            print("跳过验证集生成")
        else:
            print(f"生成完整验证集 ({len(QAS_dev)} 个样本)...")
            generate_dataset(
                num_samples=len(QAS_dev),
                save_dir=str(dev_output_path).replace('.parquet', ''),
                incremental=200,  # 每个样本使用200个文档
                qas=QAS_dev,
                docs=DOCS_dev
            )
            print(f"✓ 验证集已保存: {dev_output_path}")
    else:
        print(f"生成完整验证集 ({len(QAS_dev)} 个样本)...")
        generate_dataset(
            num_samples=len(QAS_dev),
            save_dir=str(dev_output_path).replace('.parquet', ''),
            incremental=200,  # 每个样本使用200个文档
            qas=QAS_dev,
            docs=DOCS_dev
        )
        print(f"✓ 验证集已保存: {dev_output_path}")
    
    # 验证生成的文件
    print("\n4. 验证生成的文件...")
    if train_output_path.exists():
        df_train = pd.read_parquet(train_output_path)
        print(f"✓ 训练集: {len(df_train)} 个样本")
        print(f"  文件大小: {train_output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    if dev_output_path.exists():
        df_dev = pd.read_parquet(dev_output_path)
        print(f"✓ 验证集: {len(df_dev)} 个样本")
        print(f"  文件大小: {dev_output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    print("\n" + "=" * 60)
    print("数据集生成完成！")
    print("=" * 60)
    print(f"\n训练集路径: {train_output_path}")
    print(f"验证集路径: {dev_output_path}")
    print("\n可以在训练脚本中使用这些路径:")
    print(f"  TRAIN_PATH=\"{train_output_path}\"")
    print(f"  VAL_PATH=\"{dev_output_path}\"")

if __name__ == "__main__":
    main()

