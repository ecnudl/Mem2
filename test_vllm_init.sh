#!/bin/bash
# Test script to verify vLLM can initialize without the expandable_segments error

set -x

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate memagent

# CRITICAL: Unset expandable_segments (incompatible with vLLM v1)
unset PYTORCH_CUDA_ALLOC_CONF 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF=""

# Verify the variable is not set
echo "PYTORCH_CUDA_ALLOC_CONF: '${PYTORCH_CUDA_ALLOC_CONF}'"

# Set GPU
export CUDA_VISIBLE_DEVICES=0,4,6,7
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "========================================"
echo "Testing vLLM Initialization"
echo "========================================"

# Test vLLM initialization
python3 << 'EOF'
import os
print(f"Python sees PYTORCH_CUDA_ALLOC_CONF: '{os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'NOT SET')}'")

# Check if 'expandable_segments' is in the config
conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')
if 'expandable_segments' in conf.lower():
    print("ERROR: expandable_segments is still in PYTORCH_CUDA_ALLOC_CONF!")
    print(f"Value: {conf}")
    exit(1)
else:
    print("✓ expandable_segments is not present in PYTORCH_CUDA_ALLOC_CONF")

print("\nAttempting to initialize vLLM...")
try:
    from vllm import LLM

    llm = LLM(
        model="/mnt/ssd2/models/Qwen2.5-7B-Instruct",
        tensor_parallel_size=4,
        gpu_memory_utilization=0.35,
        max_model_len=1024,
        trust_remote_code=True,
        enforce_eager=True,
    )
    print("✓ vLLM initialized successfully!")

    # Test a simple generation
    print("\nTesting generation...")
    outputs = llm.generate(["Hello, how are you?"], max_tokens=10)
    print(f"✓ Generation test passed! Output: {outputs[0].outputs[0].text[:50]}")

except Exception as e:
    print(f"✗ vLLM initialization failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n✓ All tests passed!")
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ vLLM Test PASSED"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo "✗ vLLM Test FAILED"
    echo "========================================"
    exit 1
fi
