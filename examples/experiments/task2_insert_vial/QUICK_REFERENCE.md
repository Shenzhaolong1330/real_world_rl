# Quick Reference Card

## Robot Machine Commands

### Start franka_server.py
```bash
cd /home/szl/real_world_rl/serl_robot_infra/robot_servers
python franka_server.py --flask_url 0.0.0.0
```

### Start env_server.py
```bash
cd /home/szl/real_world_rl/examples/experiments/task1_pick_banana
./run_env_server.sh
```

### Or use quick start
```bash
cd /home/szl/real_world_rl/examples/experiments/task1_pick_banana
./start_all.sh
# Choose option 1 (Robot Machine)
```

## A800 Machine Commands

### Test Connection
```bash
cd /home/szl/real_world_rl/serl_launcher
python test_pyro_env.py --ip <ROBOT_IP> --port 9090
```

### Start Learner
```bash
cd /home/szl/real_world_rl/examples/experiments/task1_pick_banana
./run_learner_conrft_remote.sh
```

### Start Actor
```bash
cd /home/szl/real_world_rl/examples/experiments/task1_pick_banana
# First, edit run_actor_conrft_remote.sh and set ROBOT_IP
./run_actor_conrft_remote.sh
```

### Or use quick start
```bash
cd /home/szl/real_world_rl/examples/experiments/task1_pick_banana
./start_all.sh
# Choose option 2 (A800 Machine)
```

## Configuration Files to Edit

### 1. Robot Machine: franka_server.py
```python
# File: serl_robot_infra/robot_servers/franka_server.py
# Line 15: Change to 0.0.0.0
parser.add_argument("--flask_url", type=str, default="0.0.0.0")
```

### 2. Robot Machine: config.py
```python
# File: examples/experiments/task1_pick_banana/config.py
# Line 15: Change to robot IP
SERVER_URL: str = "http://<ROBOT_IP>:5000/"
```

### 3. A800 Machine: run_actor_conrft_remote.sh
```bash
# File: examples/experiments/task1_pick_banana/run_actor_conrft_remote.sh
# Line 10: Change to robot IP
ROBOT_IP="192.168.1.100"  # ⚠️ CHANGE THIS
```

## Network Ports

| Port | Service | Machine | Description |
|------|---------|---------|-------------|
| 5000 | franka_server.py | Robot | Robot control (Flask) |
| 9090 | env_server.py | Robot | Remote env (Pyro5) |
| 3333 | agentlace | A800 | TrainerServer |
| 3334 | agentlace | A800 | Broadcast |

## Firewall Commands

### Robot Machine
```bash
sudo ufw allow from <A800_IP> to any port 9090
sudo ufw allow from <A800_IP> to any port 5000
```

### A800 Machine
```bash
sudo ufw allow from <A800_IP> to any port 3333
sudo ufw allow from <A800_IP> to any port 3334
```

## Troubleshooting Commands

### Check if services are running
```bash
# Robot Machine
pgrep -f franka_server.py
pgrep -f env_server.py

# A800 Machine
pgrep -f train_conrft_octo.py
```

### Check network connectivity
```bash
# From A800 to Robot
ping <ROBOT_IP>
telnet <ROBOT_IP> 9090
telnet <ROBOT_IP> 5000
```

### Monitor network traffic
```bash
# Robot Machine
sudo tcpdump -i any port 9090 -n
sudo tcpdump -i any port 5000 -n

# A800 Machine
sudo tcpdump -i any port 3333 or port 3334 -n
```

### Check logs
```bash
# Learner logs
tail -f /root/online_rl/conrft/examples/experiments/task1_pick_banana/conrft/learner.log

# Actor logs
tail -f /root/online_rl/conrft/examples/experiments/task1_pick_banana/conrft/actor.log
```

## Common Issues

### Issue: Connection Refused
```bash
# Check if env_server.py is running
pgrep -f env_server.py

# Check firewall
sudo ufw status

# Test connectivity
ping <ROBOT_IP>
```

### Issue: Import Error on A800
```bash
# This is expected! A800 should NOT have franka_env
# Solution: Use Pyro5 wrapper (--use_pyro_env flag)
```

### Issue: High Latency
```bash
# Check network bandwidth
iperf3 -c <ROBOT_IP>

# Reduce image resolution in config.py
```

## Quick Start Summary

### Robot Machine
```bash
cd /home/szl/real_world_rl/examples/experiments/task1_pick_banana
./start_all.sh  # Choose option 1
```

### A800 Machine
```bash
cd /home/szl/real_world_rl/examples/experiments/task1_pick_banana
./start_all.sh  # Choose option 2
```

That's it! 🚀
