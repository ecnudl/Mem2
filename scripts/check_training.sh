#!/bin/bash
# Check training and monitoring status

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PID_FILE="$PROJECT_ROOT/.training_pids"
TRAINING_LOG="$PROJECT_ROOT/training.log"
MONITOR_LOG="$PROJECT_ROOT/monitoring.log"
PLOTS_DIR="$PROJECT_ROOT/monitoring_plots"

echo "========================================"
echo "Training Status"
echo "========================================"
echo ""

# Check if PID file exists
if [ ! -f "$PID_FILE" ]; then
    echo "Status: NOT RUNNING (no PID file)"
    echo ""
    echo "To start training:"
    echo "  bash $SCRIPT_DIR/start_training_with_monitor.sh"
    exit 0
fi

# Read PIDs
TRAIN_PID=$(head -1 "$PID_FILE" 2>/dev/null || echo "")
MONITOR_PID=$(sed -n '2p' "$PID_FILE" 2>/dev/null || echo "")

echo "Process Status:"
echo ""

# Check training process
if [ -n "$TRAIN_PID" ]; then
    if ps -p "$TRAIN_PID" > /dev/null 2>&1; then
        TRAIN_STATUS="✓ RUNNING"
        TRAIN_MEM=$(ps -p "$TRAIN_PID" -o rss= | awk '{printf "%.1f GB", $1/1024/1024}')
        TRAIN_CPU=$(ps -p "$TRAIN_PID" -o %cpu= | awk '{printf "%.1f%%", $1}')
        TRAIN_TIME=$(ps -p "$TRAIN_PID" -o etime= | awk '{$1=$1};1')
    else
        TRAIN_STATUS="✗ NOT RUNNING"
        TRAIN_MEM="N/A"
        TRAIN_CPU="N/A"
        TRAIN_TIME="N/A"
    fi

    echo "  Training Process:"
    echo "    PID: $TRAIN_PID"
    echo "    Status: $TRAIN_STATUS"
    echo "    Memory: $TRAIN_MEM"
    echo "    CPU: $TRAIN_CPU"
    echo "    Runtime: $TRAIN_TIME"
else
    echo "  Training Process: NO PID"
fi

echo ""

# Check monitoring process
if [ -n "$MONITOR_PID" ]; then
    if ps -p "$MONITOR_PID" > /dev/null 2>&1; then
        MONITOR_STATUS="✓ RUNNING"
    else
        MONITOR_STATUS="✗ NOT RUNNING"
    fi

    echo "  Monitoring Process:"
    echo "    PID: $MONITOR_PID"
    echo "    Status: $MONITOR_STATUS"
else
    echo "  Monitoring Process: NO PID"
fi

echo ""
echo "========================================"
echo "Training Progress"
echo "========================================"
echo ""

# Check training log
if [ -f "$TRAINING_LOG" ]; then
    LOG_SIZE=$(du -h "$TRAINING_LOG" | cut -f1)
    LOG_LINES=$(wc -l < "$TRAINING_LOG")

    echo "Training Log:"
    echo "  File: $TRAINING_LOG"
    echo "  Size: $LOG_SIZE"
    echo "  Lines: $LOG_LINES"
    echo ""

    # Extract latest step info
    echo "Latest Progress:"
    LATEST_STEP=$(grep -oP 'step:\d+' "$TRAINING_LOG" | tail -1 || echo "step:0")
    if [ "$LATEST_STEP" != "step:0" ]; then
        echo "  $LATEST_STEP"

        # Try to extract latest metrics
        LATEST_LINE=$(grep "$LATEST_STEP" "$TRAINING_LOG" | tail -1)
        if echo "$LATEST_LINE" | grep -q "actor_loss"; then
            ACTOR_LOSS=$(echo "$LATEST_LINE" | grep -oP 'actor_loss:[0-9.]+' || echo "")
            REWARD=$(echo "$LATEST_LINE" | grep -oP 'reward_mean:[0-9.]+' || echo "")
            [ -n "$ACTOR_LOSS" ] && echo "  $ACTOR_LOSS"
            [ -n "$REWARD" ] && echo "  $REWARD"
        fi
    fi

    echo ""
    echo "Last 5 lines:"
    tail -5 "$TRAINING_LOG" | sed 's/^/  /'
else
    echo "Training log not found: $TRAINING_LOG"
fi

echo ""
echo "========================================"
echo "Monitoring Status"
echo "========================================"
echo ""

# Check monitoring output
if [ -f "$MONITOR_LOG" ]; then
    echo "Monitoring Log: $MONITOR_LOG"
    echo "Last update:"
    tail -3 "$MONITOR_LOG" | sed 's/^/  /'
else
    echo "Monitoring log not found: $MONITOR_LOG"
fi

echo ""

# Check plots
if [ -d "$PLOTS_DIR" ]; then
    PLOT_FILE="$PLOTS_DIR/training_curves.png"
    if [ -f "$PLOT_FILE" ]; then
        PLOT_SIZE=$(du -h "$PLOT_FILE" | cut -f1)
        PLOT_TIME=$(stat -c %y "$PLOT_FILE" | cut -d'.' -f1)
        echo "Latest Plot:"
        echo "  File: $PLOT_FILE"
        echo "  Size: $PLOT_SIZE"
        echo "  Modified: $PLOT_TIME"
    else
        echo "No plot file generated yet"
    fi
else
    echo "Plots directory not found: $PLOTS_DIR"
fi

echo ""
echo "========================================"
echo "GPU Status"
echo "========================================"
echo ""
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F', ' '{printf "  GPU %s (%s): %s%% util, %s/%s MB\n", $1, $2, $3, $4, $5}'

echo ""
echo "========================================"
echo "Quick Commands"
echo "========================================"
echo ""
echo "  # Follow training log"
echo "  tail -f $TRAINING_LOG"
echo ""
echo "  # View plot"
echo "  xdg-open $PLOTS_DIR/training_curves.png"
echo ""
echo "  # Stop training"
echo "  bash $SCRIPT_DIR/stop_training.sh"
echo ""
echo "  # Refresh status"
echo "  bash $SCRIPT_DIR/check_training.sh"
echo ""
