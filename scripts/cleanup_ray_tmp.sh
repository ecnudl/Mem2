#!/bin/bash
# Cleanup Ray temporary files to free up disk space

set -e

echo "========================================"
echo "Ray Temporary Files Cleanup"
echo "========================================"
echo ""

# Check /tmp directory usage
echo "Checking /tmp directory usage..."
TMP_USAGE=$(df -h /tmp | tail -1 | awk '{print $5}')
TMP_AVAIL=$(df -h /tmp | tail -1 | awk '{print $4}')
echo "  /tmp usage: $TMP_USAGE (available: $TMP_AVAIL)"
echo ""

# Find Ray temporary directories
echo "Finding Ray temporary directories in /tmp..."
RAY_DIRS=$(find /tmp -maxdepth 1 -type d -name "ray_*" 2>/dev/null || true)

if [ -z "$RAY_DIRS" ]; then
    echo "  No Ray directories found in /tmp"
else
    echo "  Found Ray directories:"
    for dir in $RAY_DIRS; do
        SIZE=$(du -sh "$dir" 2>/dev/null | cut -f1)
        echo "    - $dir ($SIZE)"
    done
    echo ""

    # Check if any Ray processes are running
    RAY_PROCS=$(ps aux | grep -E "ray::|raylet" | grep -v grep || true)
    if [ -n "$RAY_PROCS" ]; then
        echo "⚠️  WARNING: Ray processes are currently running!"
        echo ""
        echo "Running Ray processes:"
        echo "$RAY_PROCS"
        echo ""
        read -p "Do you want to stop all Ray processes and clean up? (yes/no): " confirm

        if [ "$confirm" == "yes" ]; then
            echo "Stopping Ray..."
            python3 -c "import ray; ray.shutdown()" 2>/dev/null || true
            pkill -9 -f "ray::" || true
            pkill -9 -f "raylet" || true
            sleep 2
            echo "Ray processes stopped."
        else
            echo "Cleanup cancelled."
            exit 0
        fi
    fi

    echo ""
    read -p "Remove all Ray temporary directories in /tmp? (yes/no): " confirm

    if [ "$confirm" == "yes" ]; then
        echo "Removing Ray temporary directories..."
        for dir in $RAY_DIRS; do
            echo "  Removing $dir..."
            rm -rf "$dir"
        done
        echo "✓ Cleanup complete!"
        echo ""

        # Show new disk usage
        echo "New /tmp directory usage:"
        TMP_USAGE=$(df -h /tmp | tail -1 | awk '{print $5}')
        TMP_AVAIL=$(df -h /tmp | tail -1 | awk '{print $4}')
        echo "  /tmp usage: $TMP_USAGE (available: $TMP_AVAIL)"
    else
        echo "Cleanup cancelled."
    fi
fi

echo ""
echo "========================================"
echo "Cleanup Complete"
echo "========================================"
