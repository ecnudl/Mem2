#!/bin/bash
# Start training and monitoring simultaneously

set -e

echo "================================================================================"
echo "Starting MemAgent Training with Real-time Monitoring"
echo "================================================================================"

# Configuration
TRAINING_SCRIPT="run_memory_7B_lora_4gpu_fast.sh"
TRAINING_LOG="training_fast.log"
MONITOR_LOG="monitor.log"
PLOT_DIR="./monitoring_plots"

# Create plot directory
mkdir -p "$PLOT_DIR"

# Clean up any previous Ray sessions
echo "Cleaning up previous sessions..."
python3 -c "import ray; ray.shutdown()" 2>/dev/null || true
pkill -f main_ppo 2>/dev/null || true
sleep 2

echo ""
echo "Step 1: Starting training in background..."
echo "  Script: $TRAINING_SCRIPT"
echo "  Log: $TRAINING_LOG"
nohup bash "$TRAINING_SCRIPT" > "$TRAINING_LOG" 2>&1 &
TRAINING_PID=$!
echo "  Training PID: $TRAINING_PID"

# Wait for log file to be created
echo ""
echo "Waiting for training log to be created..."
for i in {1..30}; do
    if [ -f "$TRAINING_LOG" ]; then
        echo "  Log file created!"
        break
    fi
    echo -n "."
    sleep 1
done

# Give training a few more seconds to start writing metrics
sleep 3

echo ""
echo "Step 2: Starting monitoring in background..."
echo "  Monitoring log: $MONITOR_LOG"
echo "  Plot directory: $PLOT_DIR"
nohup python3 scripts/monitoring/monitor_training.py \
    --mode file \
    --log-file "$TRAINING_LOG" \
    --save-dir "$PLOT_DIR" \
    --update-interval 15 \
    > "$MONITOR_LOG" 2>&1 &
MONITOR_PID=$!
echo "  Monitor PID: $MONITOR_PID"

# Save PIDs for easy stopping
echo "$TRAINING_PID" > .training.pid
echo "$MONITOR_PID" > .monitor.pid

echo ""
echo "================================================================================"
echo "Both processes started successfully!"
echo "================================================================================"
echo ""
echo "Process Information:"
echo "  Training PID: $TRAINING_PID (log: $TRAINING_LOG)"
echo "  Monitor PID:  $MONITOR_PID (log: $MONITOR_LOG)"
echo ""
echo "Monitoring Commands:"
echo "  View training log:    tail -f $TRAINING_LOG"
echo "  View monitor log:     tail -f $MONITOR_LOG"
echo "  View training plots:  ls -lh $PLOT_DIR/"
echo "  Check GPU usage:      watch -n 1 nvidia-smi"
echo ""
echo "Stop Training:"
echo "  bash scripts/stop_training.sh"
echo "  or: kill $TRAINING_PID $MONITOR_PID"
echo ""
echo "Plot will be updated every 15 seconds at:"
echo "  $PLOT_DIR/training_curves.png"
echo ""
echo "================================================================================"

# Show initial training output
echo ""
echo "Initial training output (first 50 lines):"
echo "--------------------------------------------------------------------------------"
sleep 2
head -n 50 "$TRAINING_LOG" 2>/dev/null || echo "Waiting for training to start..."

echo ""
echo "================================================================================"
echo "Setup complete! Training and monitoring are running in background."
echo "================================================================================"
