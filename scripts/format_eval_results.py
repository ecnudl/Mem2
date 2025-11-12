#!/usr/bin/env python3
"""
将验证结果转换为格式化的 JSON 文件
用法: python scripts/format_eval_results.py <jsonl_file> [output_json_file]
"""

import json
import sys
import os
from pathlib import Path
from collections import defaultdict

def format_eval_results(jsonl_path, output_json_path=None):
    """
    将 JSONL 格式的验证结果转换为格式化的 JSON
    
    Args:
        jsonl_path: 输入的 JSONL 文件路径
        output_json_path: 输出的 JSON 文件路径（可选，默认与输入文件同目录）
    """
    if not os.path.exists(jsonl_path):
        print(f"错误: 文件不存在: {jsonl_path}")
        return
    
    # 读取所有样本
    samples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    if not samples:
        print(f"警告: 文件为空: {jsonl_path}")
        return
    
    # 计算统计信息
    scores = [s.get('score', 0) for s in samples]
    total = len(samples)
    correct = sum(1 for s in scores if s > 0)
    accuracy = correct / total if total > 0 else 0.0
    avg_score = sum(scores) / total if total > 0 else 0.0
    
    # 组织结果
    result = {
        "summary": {
            "total_samples": total,
            "correct_samples": correct,
            "accuracy": accuracy,
            "average_score": avg_score,
            "checkpoint_step": samples[0].get('step', 'unknown') if samples else 'unknown',
        },
        "samples": samples,
        "statistics": {
            "score_distribution": {
                "score_0": sum(1 for s in scores if s == 0),
                "score_1": sum(1 for s in scores if s == 1),
            },
            "score_range": {
                "min": min(scores) if scores else 0,
                "max": max(scores) if scores else 0,
            }
        }
    }
    
    # 确定输出路径
    if output_json_path is None:
        jsonl_path_obj = Path(jsonl_path)
        output_json_path = jsonl_path_obj.parent / f"{jsonl_path_obj.stem}_formatted.json"
    
    # 保存结果
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 已处理 {total} 个样本")
    print(f"✓ 准确率: {accuracy:.2%} ({correct}/{total})")
    print(f"✓ 平均分数: {avg_score:.4f}")
    print(f"✓ 结果已保存到: {output_json_path}")
    
    return output_json_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python scripts/format_eval_results.py <jsonl_file> [output_json_file]")
        sys.exit(1)
    
    jsonl_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    format_eval_results(jsonl_file, output_file)

