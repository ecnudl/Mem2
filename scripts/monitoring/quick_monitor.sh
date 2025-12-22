#!/bin/bash
# Quick start script for training monitoring

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default values
EXP_NAME="lora_4gpu_balanced_20k_n8"
EXP_DIR="$PROJECT_ROOT/outputs/$EXP_NAME"
MODE="auto"
SAVE_DIR="$PROJECT_ROOT/monitoring_plots"

echo "========================================"
echo "MemAgent Training Monitor - Quick Start"
echo "========================================"
echo ""

# Check if experiment directory exists
if [ ! -d "$EXP_DIR" ]; then
    echo "Experiment directory not found: $EXP_DIR"
    echo ""
    echo "Available options:"
    echo "1. Start monitoring from WandB"
    echo "2. Start monitoring from log file"
    echo "3. Exit"
    echo ""
    read -p "Choose an option (1-3): " choice

    case $choice in
        1)
            MODE="wandb"
            read -p "Enter WandB project name [verl-memagent]: " project
            project=${project:-verl-memagent}
            read -p "Enter WandB run name [$EXP_NAME]: " run
            run=${run:-$EXP_NAME}

            echo ""
            echo "Starting WandB monitoring..."
            echo "Project: $project"
            echo "Run: $run"
            echo "Plots will be saved to: $SAVE_DIR"
            echo ""
            echo "Press Ctrl+C to stop monitoring"
            echo ""

            python3 "$SCRIPT_DIR/monitor_training.py" \
                --mode wandb \
                --project "$project" \
                --run "$run" \
                --save-dir "$SAVE_DIR"
            ;;
        2)
            read -p "Enter log file path: " logfile
            if [ ! -f "$logfile" ]; then
                echo "Error: Log file not found: $logfile"
                exit 1
            fi

            echo ""
            echo "Starting log file monitoring..."
            echo "Log file: $logfile"
            echo "Plots will be saved to: $SAVE_DIR"
            echo ""
            echo "Press Ctrl+C to stop monitoring"
            echo ""

            python3 "$SCRIPT_DIR/monitor_training.py" \
                --mode file \
                --log-file "$logfile" \
                --save-dir "$SAVE_DIR"
            ;;
        3)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo "Invalid option"
            exit 1
            ;;
    esac
else
    echo "Found experiment directory: $EXP_DIR"
    echo ""
    echo "Starting auto-detection monitoring..."
    echo "Plots will be saved to: $SAVE_DIR"
    echo ""
    echo "Press Ctrl+C to stop monitoring"
    echo ""

    python3 "$SCRIPT_DIR/monitor_training.py" \
        --mode auto \
        --exp-dir "$EXP_DIR" \
        --save-dir "$SAVE_DIR"
fi
