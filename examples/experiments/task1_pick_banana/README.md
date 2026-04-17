# Task 1: Pick Banana - Remote Deployment Guide

## Overview

This guide explains how to deploy ConRFT training across two machines:
- **Robot Machine**: Franka robot + RealSense camera (runs environment server)
- **A800 Machine**: GPU server (runs learner and actor)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Robot Machine                              │
│  ┌──────────────────┐         ┌──────────────────┐          │
│  │  franka_server.py │         │  env_server.py   │          │
│  │  (Flask HTTP)     │         │  (Pyro5 RPC)     │          │
│  │  Port: 5000       │         │  Port: 9090      │          │
│  └──────────────────┘         └──────────────────┘          │
│           ↑                            ↑                     │
└───────────┼────────────────────────────┼─────────────────────┘
            │ HTTP                        │ Pyro5 RPC
            │                             │
┌───────────┼────────────────────────────┼─────────────────────┐
│           ↓                            ↓                     │
│  ┌──────────────────────────────────────────────────┐       │
│  │              A800 Machine (GPU Server)            │       │
│  │  ┌──────────────┐         ┌──────────────┐        │       │
│  │  │   Learner    │←───────→│    Actor     │        │       │
│  │  │  (Training)  │ agentlace  (Inference) │        │       │
│  │  │  Ports:      │  3333/3334  Uses Pyro5 │        │       │
│  │  └──────────────┘         └──────────────┘        │       │
│  └──────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

### On Both Machines
- Conda environment `rl` with required packages
- Pyro5 installed: `pip install Pyro5`
- Network connectivity between machines

### On Robot Machine
- Franka robot connected and configured
- RealSense camera connected
- `franka_server.py` tested and working

### On A800 Machine
- CUDA installed and working
- GPU with sufficient memory (A800 recommended)

## Step-by-Step Deployment

### Step 1: Robot Machine Setup

1. **Start franka_server.py** (Flask HTTP server for robot control):
   ```bash
   cd /home/szl/real_world_rl/serl_robot_infra/robot_servers
   python franka_server.py --flask_url 0.0.0.0
   ```
   
   Verify it's running:
   ```bash
   curl http://localhost:5000/status
   ```

2. **Start env_server.py** (Pyro5 RPC server for environment):
   ```bash
   cd /home/szl/real_world_rl/examples/experiments/task1_pick_banana
   ./run_env_server.sh
   ```
   
   Or manually:
   ```bash
   cd /home/szl/real_world_rl/serl_robot_infra
   python env_server.py --port 9090
   ```

3. **Verify both servers are running**:
   ```bash
   # Check processes
   pgrep -f "franka_server.py"
   pgrep -f "env_server.py"
   
   # Check ports
   netstat -tlnp | grep -E "5000|9090"
   ```

4. **Note the robot machine IP**:
   ```bash
   hostname -I
   # Example output: 192.168.1.100 or 100.64.0.3
   ```

### Step 2: A800 Machine Setup

1. **Test Pyro5 connection** to robot machine:
   ```bash
   cd /home/szl/real_world_rl/serl_launcher
   python test_pyro_env.py --ip <ROBOT_IP> --port 9090
   ```
   
   Replace `<ROBOT_IP>` with the actual IP from Step 1.4.
   
   Expected output:
   ```
   ✓ Connection test passed!
   ✓ Remote environment created successfully
   ✓ Action space: Box(...)
   ✓ Observation space: Box(...)
   ```

2. **Update actor script with robot IP**:
   ```bash
   cd /home/szl/real_world_rl/examples/experiments/task1_pick_banana
   # Edit run_actor_conrft_remote.sh and set ROBOT_IP
   ```

### Step 3: Start Training

You have two options:

#### Option A: Start in Separate Terminals (Recommended)

**Terminal 1 - Start Learner**:
```bash
cd /home/szl/real_world_rl/examples/experiments/task1_pick_banana
./run_learner_conrft_remote.sh
```

Wait for learner to initialize (you'll see "Waiting for actor to connect...").

**Terminal 2 - Start Actor**:
```bash
cd /home/szl/real_world_rl/examples/experiments/task1_pick_banana
./run_actor_conrft_remote.sh
```

#### Option B: Start Both Automatically

If you have `gnome-terminal` installed:
```bash
cd /home/szl/real_world_rl/examples/experiments/task1_pick_banana
./start_all.sh
# Choose option 2 (A800 Machine)
# Choose option 3 (Both)
```

## Monitoring Training

### On A800 Machine

1. **Learner terminal** shows:
   - Training progress
   - Loss values
   - Learning rate
   - Replay buffer size

2. **Actor terminal** shows:
   - Environment interaction
   - Episode rewards
   - Action statistics

3. **Check logs**:
   ```bash
   # TensorBoard
   tensorboard --logdir /home/szl/real_world_rl/examples/experiments/task1_pick_banana/logs
   
   # Or check saved models
   ls -lh /home/szl/real_world_rl/examples/experiments/task1_pick_banana/saved_models/
   ```

### On Robot Machine

1. **franka_server.py** shows:
   - HTTP requests from actor
   - Robot state updates
   - Action commands

2. **env_server.py** shows:
   - Pyro5 RPC calls
   - Environment step/receive_action calls
   - Observation data

## Troubleshooting

### Connection Issues

**Problem**: `ConnectionRefusedError` when testing Pyro5 connection

**Solutions**:
1. Check if `env_server.py` is running on robot machine:
   ```bash
   pgrep -f "env_server.py"
   ```
2. Check firewall on robot machine:
   ```bash
   sudo ufw status
   sudo ufw allow 9090/tcp
   ```
3. Verify network connectivity:
   ```bash
   ping <ROBOT_IP>
   ```

### Serialization Errors

**Problem**: `TypeError: a bytes-like object is required, not 'dict'`

**Solution**: This is a known issue with Pyro5's serpent serializer. The fix is already applied in:
- `serl_launcher/utils/pyro_env_wrapper.py`
- `serl_robot_infra/env_server.py`

Both files use base64 encoding for numpy arrays.

### Robot Control Issues

**Problem**: Robot not responding to commands

**Solutions**:
1. Check `franka_server.py` is running:
   ```bash
   curl http://localhost:5000/status
   ```
2. Check robot hardware connection
3. Review `franka_server.py` logs for errors

### Training Not Progressing

**Problem**: Learner shows "Waiting for actor to connect..."

**Solutions**:
1. Ensure actor is started after learner
2. Check agentlace ports (3333, 3334) are not blocked
3. Verify both learner and actor are using the same experiment directory

## File Structure

```
task1_pick_banana/
├── README.md                      # This file
├── start_all.sh                   # Interactive startup script
├── run_env_server.sh              # Start Pyro5 environment server
├── run_learner_conrft_remote.sh   # Start learner on A800
├── run_actor_conrft_remote.sh     # Start actor on A800
├── config_learner_conrft_remote.gin  # Learner configuration
├── config_actor_conrft_remote.gin    # Actor configuration
├── logs/                          # Training logs (TensorBoard)
└── saved_models/                  # Saved model checkpoints
```

## Configuration Files

### config_learner_conrft_remote.gin
- Network architecture
- Learning rate schedule
- Batch size
- Training parameters

### config_actor_conrft_remote.gin
- Environment settings
- Exploration parameters
- Robot IP address
- Pyro5 connection settings

## Stopping Training

1. **Stop actor first** (Ctrl+C in actor terminal)
2. **Stop learner second** (Ctrl+C in learner terminal)
3. **Stop env_server.py** on robot machine (Ctrl+C)
4. **Stop franka_server.py** on robot machine (Ctrl+C)

## Safety Notes

⚠️ **Important**: Always ensure:
1. Robot workspace is clear before starting
2. Emergency stop is accessible
3. Robot is in compliant mode during initial testing
4. Start with low exploration noise and increase gradually

## Next Steps

After successful training:
1. Evaluate trained policy
2. Test in simulation first
3. Deploy to real robot with supervision
4. Fine-tune if necessary

## Support

For issues or questions:
1. Check logs in `logs/` directory
2. Review TensorBoard metrics
3. Consult main project README
4. Check Pyro5 documentation: https://pyro5.readthedocs.io/
