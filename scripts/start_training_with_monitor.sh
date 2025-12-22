#!/bin/bash
# Start training and monitoring in background (survives SSH disconnect)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Configuration
TRAINING_SCRIPT="$PROJECT_ROOT/run_memory_7B_lora_4gpu_balanced.sh"
TRAINING_LOG="$PROJECT_ROOT/training.log"
MONITOR_LOG="$PROJECT_ROOT/monitoring.log"
MONITOR_PLOTS="$PROJECT_ROOT/monitoring_plots"
PID_FILE="$PROJECT_ROOT/.training_pids"

echo "========================================"
echo "MemAgent Background Training Launcher"
echo "========================================"
echo ""

# Check if training is already running
if [ -f "$PID_FILE" ]; then
    TRAIN_PID=$(head -1 "$PID_FILE" 2>/dev/null || echo "")
    if [ -n "$TRAIN_PID" ] && ps -p "$TRAIN_PID" > /dev/null 2>&1; then
        echo "⚠️  Training is already running (PID: $TRAIN_PID)"
        echo ""
        read -p "Stop existing training and restart? (yes/no): " confirm
        if [ "$confirm" != "yes" ]; then
            echo "Cancelled."
            exit 0
        fi
        echo "Stopping existing training..."
        bash "$SCRIPT_DIR/stop_training.sh"
        sleep 2
    fi
fi

# Check if training script exists
if [ ! -f "$TRAINING_SCRIPT" ]; then
    echo "Error: Training script not found: $TRAINING_SCRIPT"
    exit 1
fi

# Clean up old log files (optional)
read -p "Clean up old logs? (yes/no) [no]: " clean_logs
clean_logs=${clean_logs:-no}
if [ "$clean_logs" == "yes" ]; then
    echo "Cleaning up old logs..."
    rm -f "$TRAINING_LOG" "$MONITOR_LOG"
    rm -rf "$MONITOR_PLOTS"
fi

echo ""
echo "Starting training in background..."
echo "  Training script: $TRAINING_SCRIPT"
echo "  Training log: $TRAINING_LOG"
echo ""

# Start training in background with nohup
cd "$PROJECT_ROOT"
nohup bash "$TRAINING_SCRIPT" > "$TRAINING_LOG" 2>&1 &
TRAIN_PID=$!

echo "✓ Training started (PID: $TRAIN_PID)"

# Wait for training log to be created and have some content
echo "Waiting for training to initialize..."
for i in {1..30}; do
    if [ -f "$TRAINING_LOG" ] && [ -s "$TRAINING_LOG" ]; then
        echo "✓ Training log created"
        break
    fi
    sleep 1
    echo -n "."
done
echo ""

# Give it a few more seconds to ensure logging is stable
sleep 3

# Check if training process is still running
if ! ps -p "$TRAIN_PID" > /dev/null 2>&1; then
    echo "✗ Training process failed to start!"
    echo ""
    echo "Last 20 lines of training log:"
    tail -20 "$TRAINING_LOG"
    exit 1
fi

echo ""
echo "Starting monitoring in background..."
echo "  Monitor log: $MONITOR_LOG"
echo "  Plots directory: $MONITOR_PLOTS"
echo ""

# Start monitoring in background
cd "$SCRIPT_DIR/monitoring"
nohup python3 monitor_training.py \
    --mode file \
    --log-file "$TRAINING_LOG" \
    --save-dir "$MONITOR_PLOTS" \
    --update-interval 30 \
    > "$MONITOR_LOG" 2>&1 &
MONITOR_PID=$!

echo "✓ Monitoring started (PID: $MONITOR_PID)"

# Save PIDs
echo "$TRAIN_PID" > "$PID_FILE"
echo "$MONITOR_PID" >> "$PID_FILE"

echo ""
echo "========================================"
echo "✓ Training and Monitoring Started!"
echo "========================================"
echo ""
echo "Process Information:"
echo "  Training PID: $TRAIN_PID"
echo "  Monitor PID: $MONITOR_PID"
echo ""
echo "Logs:"
echo "  Training: $TRAINING_LOG"
echo "  Monitor: $MONITOR_LOG"
echo ""
echo "Output:"
echo "  Plots: $MONITOR_PLOTS/training_curves.png"
echo ""
echo "Useful Commands:"
echo "  # Check training progress"
echo "  tail -f $TRAINING_LOG"
echo ""
echo "  # Check monitoring status"
echo "  tail -f $MONITOR_LOG"
echo ""
echo "  # View latest plot"
echo "  xdg-open $MONITOR_PLOTS/training_curves.png"
echo ""
echo "  # Stop everything"
echo "  bash $SCRIPT_DIR/stop_training.sh"
echo ""
echo "  # Check if processes are running"
echo "  ps -p $TRAIN_PID,$MONITOR_PID"
echo ""
echo "These processes will continue running even if you:"
echo "  - Close VSCode"
echo "  - Disconnect SSH"
echo "  - Close terminal"
echo ""
echo "To stop training, run: bash $SCRIPT_DIR/stop_training.sh"
echo ""
