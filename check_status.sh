#!/bin/bash
# Quick status check for training and monitoring

echo "================================================================================"
echo "MemAgent Training Status - $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"
echo ""

# Check processes
echo "Process Status:"
TRAINING_PID=$(cat .training.pid 2>/dev/null)
MONITOR_PID=$(cat .monitor.pid 2>/dev/null)

if [ -n "$TRAINING_PID" ] && ps -p $TRAINING_PID > /dev/null 2>&1; then
    echo "  ✓ Training running (PID: $TRAINING_PID)"
else
    echo "  ✗ Training not running"
fi

if [ -n "$MONITOR_PID" ] && ps -p $MONITOR_PID > /dev/null 2>&1; then
    echo "  ✓ Monitor running (PID: $MONITOR_PID)"
else
    echo "  ✗ Monitor not running"
fi

echo ""
echo "GPU Usage:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader | \
    awk -F, '{printf "  GPU %s: %s / %s (Util: %s)\n", $1, $3, $4, $5}'

echo ""
echo "Recent Training Output:"
echo "--------------------------------------------------------------------------------"
tail -n 5 training_fast.log 2>/dev/null | grep -E "(step:|epoch:|reward|loss)" || echo "  No training metrics yet..."

echo ""
echo "Latest Plot:"
if [ -f "monitoring_plots/training_curves.png" ]; then
    PLOT_TIME=$(stat -c %y monitoring_plots/training_curves.png)
    PLOT_SIZE=$(du -h monitoring_plots/training_curves.png | cut -f1)
    echo "  File: monitoring_plots/training_curves.png"
    echo "  Size: $PLOT_SIZE"
    echo "  Last updated: $PLOT_TIME"
else
    echo "  No plot generated yet"
fi

echo ""
echo "================================================================================"
echo "Commands:"
echo "  View training log:  tail -f training_fast.log"
echo "  View monitor log:   tail -f monitor.log"
echo "  View GPU live:      watch -n 1 nvidia-smi"
echo "  Stop training:      bash scripts/stop_training.sh"
echo "================================================================================"
