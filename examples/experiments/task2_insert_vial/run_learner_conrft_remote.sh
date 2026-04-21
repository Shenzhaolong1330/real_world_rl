#!/bin/bash
# Run this on A800 machine
# This script starts the LEARNER (no hardware dependencies needed)

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5

# Configuration
A800_IP="localhost"  # A800's IP (learner binds to localhost)
CHECKPOINT_PATH="/home/szl/real_world_rl/examples/experiments/task2_insert_vial/conrft"
DEMO_PATH="/home/szl/real_world_rl/demo_data/task2_insert_vial_30_demos_2026-04-21_08-57-14.pkl"

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
    --exp_name=task2_insert_vial \
    --checkpoint_path=$CHECKPOINT_PATH \
    --q_weight=1.0 \
    --bc_weight=0.1 \
    --demo_path=$DEMO_PATH \
    --pretrain_steps=5000 \
    --debug=False \
    --learner \
    --ip=$A800_IP