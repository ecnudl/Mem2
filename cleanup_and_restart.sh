#!/bin/bash
# Cleanup and Restart Script for LoRA Training
# Use this script to clean up old processes and restart training after OOM

set -x

echo "========================================"
echo "Cleaning up old processes and temp files"
echo "========================================"

# 1. Kill any existing Ray processes
echo "Killing Ray processes..."
pkill -9 -f "ray::" 2>/dev/null || true
pkill -9 -f "raylet" 2>/dev/null || true

# 2. Clean up Ray cluster
echo "Shutting down Ray..."
python3 -c "import ray; ray.shutdown()" 2>/dev/null || true

# 3. Wait a bit for processes to clean up
sleep 3

# 4. Clean up Ray temp directories (optional - only if disk space is low)
read -p "Clean up Ray temp directories? This will free disk space but may take time. (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Cleaning Ray temp directories..."
    rm -rf /tmp/ray_local__* 2>/dev/null || true
    rm -rf ~/ray_tmp_lora 2>/dev/null || true
    rm -rf /home/admin123/dl/MemAgent/outputs/ray_tmp_lora/* 2>/dev/null || true
fi

# 5. Check GPU status
echo ""
echo "========================================"
echo "Current GPU Status:"
echo "========================================"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
  awk -F, '{printf "GPU %s: %s/%s MB (%.1f%% util)\n", $1, $2, $3, $4}'

echo ""
echo "Checking GPUs 4,5,6,7 (training GPUs)..."
for gpu in 4 5 6 7; do
    mem_used=$(nvidia-smi -i $gpu --query-gpu=memory.used --format=csv,noheader,nounits)
    mem_total=$(nvidia-smi -i $gpu --query-gpu=memory.total --format=csv,noheader,nounits)
    mem_free=$((mem_total - mem_used))

    if [ $mem_free -lt 40000 ]; then
        echo "WARNING: GPU $gpu has only ${mem_free}MB free (need ~40GB for training)"
        echo "There may be other processes using this GPU:"
        nvidia-smi -i $gpu | grep -A 3 "Processes:"
    else
        echo "GPU $gpu: OK - ${mem_free}MB free"
    fi
done

echo ""
echo "========================================"
echo "Cleanup complete!"
echo "========================================"
echo ""
echo "Ready to restart training. Run:"
echo "  bash run_memory_7B_lora.sh"
echo ""
