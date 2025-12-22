#!/bin/bash
# Stop background training and monitoring

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PID_FILE="$PROJECT_ROOT/.training_pids"

echo "========================================"
echo "Stop Training and Monitoring"
echo "========================================"
echo ""

if [ ! -f "$PID_FILE" ]; then
    echo "No PID file found. Training may not be running."
    echo ""
    echo "Checking for any Python training processes..."

    # Try to find training processes
    TRAIN_PROCS=$(ps aux | grep "verl.trainer.main_ppo" | grep -v grep || true)
    MONITOR_PROCS=$(ps aux | grep "monitor_training.py" | grep -v grep || true)

    if [ -n "$TRAIN_PROCS" ]; then
        echo "Found training processes:"
        echo "$TRAIN_PROCS"
        echo ""
        read -p "Kill these processes? (yes/no): " confirm
        if [ "$confirm" == "yes" ]; then
            pkill -f "verl.trainer.main_ppo" || true
            echo "✓ Training processes killed"
        fi
    fi

    if [ -n "$MONITOR_PROCS" ]; then
        echo "Found monitoring processes:"
        echo "$MONITOR_PROCS"
        echo ""
        read -p "Kill these processes? (yes/no): " confirm
        if [ "$confirm" == "yes" ]; then
            pkill -f "monitor_training.py" || true
            echo "✓ Monitoring processes killed"
        fi
    fi

    if [ -z "$TRAIN_PROCS" ] && [ -z "$MONITOR_PROCS" ]; then
        echo "No training or monitoring processes found."
    fi

    exit 0
fi

# Read PIDs
TRAIN_PID=$(head -1 "$PID_FILE" 2>/dev/null || echo "")
MONITOR_PID=$(sed -n '2p' "$PID_FILE" 2>/dev/null || echo "")

echo "Stored PIDs:"
echo "  Training: $TRAIN_PID"
echo "  Monitor: $MONITOR_PID"
echo ""

STOPPED=0

# Stop training
if [ -n "$TRAIN_PID" ]; then
    if ps -p "$TRAIN_PID" > /dev/null 2>&1; then
        echo "Stopping training (PID: $TRAIN_PID)..."
        kill "$TRAIN_PID" 2>/dev/null || true

        # Wait for graceful shutdown
        for i in {1..10}; do
            if ! ps -p "$TRAIN_PID" > /dev/null 2>&1; then
                break
            fi
            sleep 1
            echo -n "."
        done
        echo ""

        # Force kill if still running
        if ps -p "$TRAIN_PID" > /dev/null 2>&1; then
            echo "Force killing training..."
            kill -9 "$TRAIN_PID" 2>/dev/null || true
        fi

        echo "✓ Training stopped"
        STOPPED=1
    else
        echo "Training process (PID: $TRAIN_PID) not running"
    fi
fi

# Stop monitoring
if [ -n "$MONITOR_PID" ]; then
    if ps -p "$MONITOR_PID" > /dev/null 2>&1; then
        echo "Stopping monitoring (PID: $MONITOR_PID)..."
        kill "$MONITOR_PID" 2>/dev/null || true
        sleep 1

        # Force kill if still running
        if ps -p "$MONITOR_PID" > /dev/null 2>&1; then
            kill -9 "$MONITOR_PID" 2>/dev/null || true
        fi

        echo "✓ Monitoring stopped"
        STOPPED=1
    else
        echo "Monitoring process (PID: $MONITOR_PID) not running"
    fi
fi

# Clean up Ray processes
echo ""
echo "Cleaning up Ray processes..."
python3 -c "import ray; ray.shutdown()" 2>/dev/null || true
pkill -f "ray::" 2>/dev/null || true
pkill -f "raylet" 2>/dev/null || true

# Remove PID file
rm -f "$PID_FILE"

echo ""
if [ $STOPPED -eq 1 ]; then
    echo "========================================"
    echo "✓ All Processes Stopped"
    echo "========================================"
else
    echo "========================================"
    echo "No Active Processes Found"
    echo "========================================"
fi
echo ""
echo "Logs are preserved:"
echo "  Training: $PROJECT_ROOT/training.log"
echo "  Monitor: $PROJECT_ROOT/monitoring.log"
echo "  Plots: $PROJECT_ROOT/monitoring_plots/"
echo ""
