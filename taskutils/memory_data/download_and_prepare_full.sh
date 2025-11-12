#!/bin/bash
# 下载并准备完整的HotpotQA数据集
# 用法: bash download_and_prepare_full.sh

set -e  # 遇到错误立即退出

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "HotpotQA完整数据集下载和准备脚本"
echo "=========================================="
echo ""

# 1. 检查并安装依赖
echo "1. 检查依赖..."
if ! command -v aria2c &> /dev/null; then
    echo "  安装 aria2c..."
    sudo apt update
    yes | sudo apt install aria2 || {
        echo "错误: 无法安装 aria2c，请手动安装"
        exit 1
    }
fi

if ! python3 -c "import pandas" 2>/dev/null; then
    echo "  安装 Python 依赖..."
    pip3 install pandas pyarrow || {
        echo "错误: 无法安装 Python 依赖"
        exit 1
    }
fi

# 2. 下载原始数据
echo ""
echo "2. 下载原始数据..."
if [ ! -f "hotpotqa_train.json" ] || [ ! -f "hotpotqa_dev.json" ]; then
    echo "  下载 HotpotQA 数据集..."
    
    # 创建下载列表
    cat > __download.txt << EOF
http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
    out=hotpotqa_train.json
http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
    out=hotpotqa_dev.json
EOF
    
    aria2c -x 10 -s 10 -j 2 -i __download.txt || {
        echo "错误: 下载失败"
        rm -f __download.txt
        exit 1
    }
    
    rm -f __download.txt
    echo "  ✓ 下载完成"
else
    echo "  ✓ 数据文件已存在，跳过下载"
fi

# 3. 生成完整数据集
echo ""
echo "3. 生成完整数据集..."
echo "   这将生成:"
echo "   - 训练集: 32k 样本 -> hotpotqa/hotpotqa_train_32k.parquet"
echo "   - 验证集: 完整验证集 -> hotpotqa/hotpotqa_dev.parquet"
echo ""

python3 prepare_full_dataset.py || {
    echo "错误: 数据集生成失败"
    exit 1
}

echo ""
echo "=========================================="
echo "完成！数据集已准备就绪"
echo "=========================================="
echo ""
echo "数据集路径:"
echo "  训练集: $SCRIPT_DIR/hotpotqa/hotpotqa_train_32k.parquet"
echo "  验证集: $SCRIPT_DIR/hotpotqa/hotpotqa_dev.parquet"
echo ""
echo "在训练脚本中使用:"
echo "  DATASET_ROOT=$SCRIPT_DIR"
echo "  TRAIN_PATH=\${DATASET_ROOT}/hotpotqa/hotpotqa_train_32k.parquet"
echo "  VAL_PATH=\${DATASET_ROOT}/hotpotqa/hotpotqa_dev.parquet"

