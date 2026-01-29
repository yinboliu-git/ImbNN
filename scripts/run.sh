#!/bin/bash

# ============================================================
#  Robust Dynamic GPU Job Queue
# ============================================================

# --- Configuration ---
NUM_GPUS=4
TASKS_PER_GPU=3
# --------------------

# 1. Basic setup: Create timestamped experiment root directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_ROOT="experiments/$TIMESTAMP"
CONFIG_DIR="$EXP_ROOT/configs"
LOG_DIR="$EXP_ROOT/logs"
RESULT_DIR="$EXP_ROOT/results"

mkdir -p "$CONFIG_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$RESULT_DIR"

echo "========================================================"
echo "Starting Experiment Batch: $TIMESTAMP"
echo "Root Directory: $EXP_ROOT"
echo "========================================================"

# 2. Safe signal handling
trap 'echo ">>> Caught Signal! Killing jobs..."; kill $(jobs -p) 2>/dev/null; rm -f $FIFO_FILE; exit 1' SIGINT SIGTERM EXIT

# 3. Generate configuration files
echo ">>> Generating configs..."

# Clean up any residual configs directory
rm -rf configs

# Run Python script (generates configs/exp1/, configs/exp2/, etc.)
python gen_configs.py

# Check if generation succeeded
if [ ! -d "configs" ]; then
    echo "Error: 'configs' directory not generated! Check gen_configs.py."
    exit 1
fi

echo ">>> Moving generated configs to: $CONFIG_DIR"

# Recursive copy to preserve exp1, exp2 subdirectory structure
cp -r configs/* "$CONFIG_DIR/"

# Check target folder
NUM_FILES=$(find "$CONFIG_DIR" -name "*.yaml" | wc -l)

if [ "$NUM_FILES" -eq 0 ]; then
    echo "Error: No .yaml files found in $CONFIG_DIR!"
    exit 1
fi

# Clean up temporary configs folder
rm -rf configs

echo "Ready: Found $NUM_FILES tasks across sub-directories."

# 4. Initialize token bucket
FIFO_FILE="/tmp/$$.fifo"
mkfifo $FIFO_FILE
exec 6<>$FIFO_FILE
rm $FIFO_FILE

# Fill tokens
for ((i=0; i<NUM_GPUS; i++)); do
    for ((j=0; j<TASKS_PER_GPU; j++)); do
        echo "$i" >&6
    done
done

# 5. Dynamic task distribution
echo ">>> Starting Queue Execution..."
count=0

# Find recursively searches exp1/*.yaml, exp2/*.yaml
exec 3< <(find "$CONFIG_DIR" -name "*.yaml" | sort)

while read -u3 config_path; do
    ((count++))
    EXP_NAME=$(basename "$config_path" .yaml)

    # Get GPU token
    read -u6 GPU_ID

    echo "[Progress $count/$NUM_FILES] Launching $EXP_NAME on GPU $GPU_ID ..."

    {
        # --- Subprocess ---
        # 1. Run task
        CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
            --config "$config_path" \
            --save_root "$RESULT_DIR" \
            > "$LOG_DIR/${EXP_NAME}.log" 2>&1

        # 2. Return token
        echo "$GPU_ID" >&6
    } &
done

# Close fd3
exec 3<&-

# Wait for completion
wait
echo "========================================================"
echo "All Jobs Finished!"
echo "Review Results at: $EXP_ROOT"
echo "========================================================"
