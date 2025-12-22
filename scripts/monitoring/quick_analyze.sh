#!/bin/bash
# Quick start script for post-training analysis

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default values
EXP_NAME="lora_4gpu_balanced_20k_n8"
EXP_DIR="$PROJECT_ROOT/outputs/$EXP_NAME"
OUTPUT_DIR="$PROJECT_ROOT/analysis_results"

echo "============================================="
echo "MemAgent Training Analyzer - Quick Start"
echo "============================================="
echo ""

# Check if experiment directory exists
if [ ! -d "$EXP_DIR" ]; then
    echo "Experiment directory not found: $EXP_DIR"
    echo ""
    echo "Available options:"
    echo "1. Analyze from WandB"
    echo "2. Analyze from log file"
    echo "3. Exit"
    echo ""
    read -p "Choose an option (1-3): " choice

    case $choice in
        1)
            read -p "Enter WandB project name [verl-memagent]: " project
            project=${project:-verl-memagent}
            read -p "Enter WandB run name [$EXP_NAME]: " run
            run=${run:-$EXP_NAME}

            echo ""
            echo "Analyzing from WandB..."
            echo "Project: $project"
            echo "Run: $run"
            echo "Results will be saved to: $OUTPUT_DIR"
            echo ""

            python3 "$SCRIPT_DIR/analyze_training.py" \
                --source wandb \
                --project "$project" \
                --run "$run" \
                --output-dir "$OUTPUT_DIR"
            ;;
        2)
            read -p "Enter log file path: " logfile
            if [ ! -f "$logfile" ]; then
                echo "Error: Log file not found: $logfile"
                exit 1
            fi

            echo ""
            echo "Analyzing from log file..."
            echo "Log file: $logfile"
            echo "Results will be saved to: $OUTPUT_DIR"
            echo ""

            python3 "$SCRIPT_DIR/analyze_training.py" \
                --source file \
                --log-file "$logfile" \
                --output-dir "$OUTPUT_DIR"
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
    echo "Starting auto-detection analysis..."
    echo "Results will be saved to: $OUTPUT_DIR"
    echo ""

    python3 "$SCRIPT_DIR/analyze_training.py" \
        --source auto \
        --exp-dir "$EXP_DIR" \
        --output-dir "$OUTPUT_DIR"
fi

echo ""
echo "============================================="
echo "Analysis Complete!"
echo "============================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
echo "  - comprehensive_analysis.png  (9-plot comprehensive analysis)"
echo "  - training_report.json       (detailed metrics report)"
echo ""
echo "View the analysis:"
if command -v xdg-open &> /dev/null; then
    echo "  xdg-open $OUTPUT_DIR/comprehensive_analysis.png"
elif command -v open &> /dev/null; then
    echo "  open $OUTPUT_DIR/comprehensive_analysis.png"
else
    echo "  Use your preferred image viewer to open the PNG file"
fi
echo ""
