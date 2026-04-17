#!/bin/bash
# Run this on A800 machine
# This script starts the LEARNER (no hardware dependencies needed)

# Set environment variables
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5

# Configuration
A800_IP="localhost"  # A800's IP (learner binds to localhost)
CHECKPOINT_PATH="/root/online_rl/conrft/examples/experiments/task1_pick_banana/conrft"
DEMO_PATH="./demo_data/task1_pick_banana_30_demos.pkl"

echo "=========================================="
echo "Starting LEARNER on A800"
echo "=========================================="
echo "IP: $A800_IP"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Demo: $DEMO_PATH"
echo "=========================================="

# Start learner (fake_env=True automatically)
cd /home/szl/real_world_rl/examples
python train_conrft_octo.py \
    --exp_name=task1_pick_banana \
    --checkpoint_path=$CHECKPOINT_PATH \
    --q_weight=1.0 \
    --bc_weight=0.1 \
    --demo_path=$DEMO_PATH \
    --pretrain_steps=20000 \
    --debug=False \
    --learner \
    --ip=$A800_IP
