#!/bin/bash
# Run this on the ROBOT machine (Franka + RealSense)
# This script starts the Pyro5 environment server

# Set environment variables
export PYTHONPATH=/home/szl/real_world_rl:$PYTHONPATH

# Configuration
LOCAL_IP="0.0.0.0"  # Listen on all interfaces
PYRO_PORT=9090
TASK_NAME="task2_close_cap"

echo "=========================================="
echo "Starting Pyro5 Environment Server"
echo "=========================================="
echo "Task: $TASK_NAME"
echo "Host: $LOCAL_IP"
echo "Port: $PYRO_PORT"
echo "=========================================="

# Start Pyro5 environment server
cd /home/szl/real_world_rl/serl_robot_infra
python env_server.py \
    --host $LOCAL_IP \
    --port $PYRO_PORT \
    --task $TASK_NAME
