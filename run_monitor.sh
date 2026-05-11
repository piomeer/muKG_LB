#!/bin/bash

# Remove old log
rm -f gpu_monitor.log

# Start nvidia-smi dmon monitoring SM and memory utilization in background
# -s u: shows sm, mem, enc, dec utilization
# -d 1: update every 1 second
nvidia-smi dmon -s u -d 1 > gpu_monitor.log 2>&1 &
MONITOR_PID=$!
echo "GPU monitor started (PID: $MONITOR_PID)"

# Run training script (20 steps)
python profile_transe.py
echo "Training script finished."

# Kill the GPU monitor
kill $MONITOR_PID 2>/dev/null
wait $MONITOR_PID 2>/dev/null
echo "GPU monitor stopped."

# Show the log
echo ""
echo "========== GPU Monitor Log Summary =========="
cat gpu_monitor.log
