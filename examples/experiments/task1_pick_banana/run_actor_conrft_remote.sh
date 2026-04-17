#!/bin/bash
# Run this on A800 machine
# This script starts the ACTOR (connects to remote env via Pyro5)

# Set environment variables
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2

# Configuration - YOU NEED TO MODIFY THESE
A800_IP="localhost"  # A800's IP (learner is running here)
ROBOT_IP="192.168.1.100"  # ⚠️ CHANGE THIS to your robot machine IP
PYRO_PORT=9090
CHECKPOINT_PATH="/root/online_rl/conrft/examples/experiments/task1_pick_banana/conrft"

echo "=========================================="
echo "Starting ACTOR on A800"
echo "=========================================="
echo "Learner IP: $A800_IP"
echo "Robot IP: $ROBOT_IP"
echo "Pyro5 Port: $PYRO_PORT"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "=========================================="

# Start actor with Pyro5 remote environment
cd /home/szl/real_world_rl/examples
python train_conrft_octo.py \
    --exp_name=task1_pick_banana \
    --checkpoint_path=$CHECKPOINT_PATH \
    --actor \
    --ip=$A800_IP \
    --use_pyro_env \
    --pyro_env_ip=$ROBOT_IP \
    --pyro_env_port=$PYRO_PORT
