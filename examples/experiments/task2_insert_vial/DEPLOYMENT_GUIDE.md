# Remote Deployment Guide: A800 + Franka Robot

## Architecture

```
Robot Machine (Franka + RealSense)          A800 Machine (GPU Server)
┌─────────────────────────────────┐        ┌──────────────────────────────┐
│  franka_server.py (Flask:5000)  │        │  learner (fake_env=True)      │
│  env_server.py (Pyro5:9090)     │◄───────┤  actor (Pyro5 client)         │
│  - Real environment             │        │  - No hardware dependencies   │
│  - Hardware access              │        │  - JAX/GPU training           │
└─────────────────────────────────┘        └──────────────────────────────┘
         │                                          │
         │                                          │
         └──────────── agentlace (3333/3334) ───────┘
              (actor ↔ learner parameter sync)
```

## Step-by-Step Deployment

### Step 1: Robot Machine Setup

#### 1.1 Install Pyro5
```bash
pip install Pyro5
```

#### 1.2 Configure franka_server.py for Remote Access

Edit `serl_robot_infra/robot_servers/franka_server.py`:
```python
# Line 15: Change from "127.0.0.1" to "0.0.0.0"
parser.add_argument("--flask_url", type=str, default="0.0.0.0")
```

#### 1.3 Configure config.py for Remote Access

Edit `examples/experiments/task1_pick_banana/config.py`:
```python
class EnvConfig(DefaultEnvConfig):
    # Change to robot machine IP
    SERVER_URL: str = "http://<ROBOT_MACHINE_IP>:5000/"
    # Example: SERVER_URL = "http://192.168.1.100:5000/"
```

#### 1.4 Start Servers on Robot Machine

**Terminal 1: Start franka_server.py**
```bash
cd /home/szl/real_world_rl/serl_robot_infra/robot_servers
python franka_server.py --flask_url 0.0.0.0
```

Expected output:
```
 * Running on http://0.0.0.0:5000
 * Restarting with stat
 * Debugger is active!
```

**Terminal 2: Start env_server.py**
```bash
cd /home/szl/real_world_rl/examples/experiments/task1_pick_banana
chmod +x run_env_server.sh
./run_env_server.sh
```

Expected output:
```
==========================================
Starting Pyro5 Environment Server
==========================================
Task: task1_pick_banana
Host: 0.0.0.0
Port: 9090
==========================================

Creating environment for task: task1_pick_banana
Environment created successfully
Action space: Box(-1.0, 1.0, (7,), float32)
Observation space: Dict(...)

============================================================
Pyro5 Environment Server Started
============================================================
URI: PYRO:remote_env@0.0.0.0:9090
Host: 0.0.0.0
Port: 9090

On A800 machine, use this URI to connect:
  PYRO:remote_env@<ROBOT_MACHINE_IP>:9090
============================================================
```

### Step 2: A800 Machine Setup

#### 2.1 Install Pyro5
```bash
pip install Pyro5
```

#### 2.2 Install serl_launcher (without hardware dependencies)
```bash
cd /home/szl/real_world_rl/serl_launcher
pip install -e .
```

**⚠️ Important**: A800 does NOT need `serl_robot_infra` or `franka_env`!

#### 2.3 Configure Scripts

Edit `run_learner_conrft_remote.sh`:
```bash
# No changes needed, uses localhost by default
```

Edit `run_actor_conrft_remote.sh`:
```bash
# Change ROBOT_IP to your robot machine IP
ROBOT_IP="192.168.1.100"  # ⚠️ CHANGE THIS
```

#### 2.4 Test Connection

Before starting training, test the Pyro5 connection:

```bash
cd /home/szl/real_world_rl/serl_launcher
python test_pyro_env.py --ip <ROBOT_MACHINE_IP> --port 9090
```

Expected output:
```
Testing Pyro5 Environment Connection
============================================================

Connecting to remote environment at <ROBOT_MACHINE_IP>:9090...
Connected to remote environment at PYRO:remote_env@<ROBOT_MACHINE_IP>:9090
Remote action space: Box(-1.0, 1.0, (7,), float32)
Remote observation space: Dict(...)

[Test 1] Action Space:
  Shape: (7,)
  Low: [-1. -1. -1. -1. -1. -1. -1.]
  High: [1. 1. 1. 1. 1. 1.1.]

[Test 2] Observation Space:
  state: shape=(15,), dtype=float32
  images: shape=(2, 84, 84), dtype=uint8
  ...

All tests passed!
```

### Step 3: Start Training

#### 3.1 Start Learner (on A800)

**Terminal 1:**
```bash
cd /home/szl/real_world_rl/examples/experiments/task1_pick_banana
chmod +x run_learner_conrft_remote.sh
./run_learner_conrft_remote.sh
```

Expected output:
```
==========================================
Starting LEARNER on A800
==========================================
IP: localhost
Checkpoint: /root/online_rl/conrft/examples/experiments/task1_pick_banana/conrft
Demo: ./demo_data/task1_pick_banana_30_demos.pkl
==========================================

Pretraining the model with demo data
pretraining: 100%|██████████| 20000/20000 [10:23<00:00, 32.11it/s]
Pretraining done
sent initial network to actor
Filling up replay buffer: 100%|██████████| 1000/1000 [00:05<00:00, 180.25it/s]
learner: 0%|          | 0/100000 [00:00<?, ?it/s]
```

#### 3.2 Start Actor (on A800)

**Terminal 2:**
```bash
cd /home/szl/real_world_rl/examples/experiments/task1_pick_banana
chmod +x run_actor_conrft_remote.sh
./run_actor_conrft_remote.sh
```

Expected output:
```
==========================================
Starting ACTOR on A800
==========================================
Learner IP: localhost
Robot IP: 192.168.1.100
Pyro5 Port: 9090
Checkpoint: /root/online_rl/conrft/examples/experiments/task1_pick_banana/conrft
==========================================

Connecting to Pyro5 remote environment at 192.168.1.100:9090...
Connected to remote environment at PYRO:remote_env@192.168.1.100:9090
Remote action space: Box(-1.0, 1.0, (7,), float32)
Remote observation space: Dict(...)

starting actor loop
  0%|          | 0/100000 [00:00<?, ?it/s]
```

## Network Configuration

### Firewall Rules

**Robot Machine:**
```bash
# Allow A800 to access Pyro5 server (port 9090)
sudo ufw allow from <A800_IP> to any port 9090

# Allow A800 to access franka server (port 5000)
sudo ufw allow from <A800_IP> to any port 5000
```

**A800 Machine:**
```bash
# Allow actor to connect to learner (ports 3333, 3334)
sudo ufw allow from <A800_IP> to any port 3333
sudo ufw allow from <A800_IP> to any port 3334
```

### Port Summary

| Port | Service | Machine | Direction | Description |
|------|---------|---------|-----------|-------------|
| 5000 | franka_server.py | Robot | Robot → Robot | Flask HTTP for robot control |
| 9090 | env_server.py | Robot | Robot → A800 | Pyro5 remote env calls |
| 3333 | agentlace | A800 | A800 ↔ A800 | TrainerServer (learner) |
| 3334 | agentlace | A800 | A800 → A800 | Broadcast (learner → actor) |

## Troubleshooting

### Issue 1: Connection Refused

**Error:**
```
Pyro5.errors.ConnectionRefusedError: Cannot connect to remote_env@<ROBOT_IP>:9090
```

**Solution:**
1. Check if `env_server.py` is running on robot machine
2. Check firewall rules: `sudo ufw status`
3. Test connectivity: `ping <ROBOT_IP>`
4. Verify IP address: `ifconfig` or `ip addr`

### Issue 2: Import Error on A800

**Error:**
```
ModuleNotFoundError: No module named 'franka_env'
```

**Solution:**
This is expected! A800 should NOT have `franka_env` installed. The Pyro5 wrapper avoids this dependency.

### Issue 3: High Latency

**Warning:**
```
Warning: Step took >100ms
```

**Solution:**
1. Check network bandwidth: `iperf3 -c <ROBOT_IP>`
2. Reduce image resolution in config
3. Use wired Ethernet instead of WiFi

### Issue 4: Demo Data Not Found

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: './demo_data/task1_pick_banana_30_demos.pkl'
```

**Solution:**
1. Copy demo data to A800:
   ```bash
   scp robot@<ROBOT_IP>:/path/to/demo_data/task1_pick_banana_30_demos.pkl ./demo_data/
   ```
2. Or update `DEMO_PATH` in `run_learner_conrft_remote.sh`

## Performance Optimization

### Network Latency
- Typical Pyro5 call: 10-50ms
- Image transfer: 5-20ms (depends on resolution)
- Total overhead: ~30-70ms per step

### Bandwidth Usage
- Observation size: ~50KB (state + compressed images)
- Action size: ~28 bytes
- Total: ~50KB per step × 10Hz = 500KB/s

### Tips
1. Use compressed images (JPEG/PNG) in observation
2. Reduce image resolution if needed (84x84 is usually enough)
3. Use wired Ethernet instead of WiFi

## Monitoring

### Check Logs

**Learner logs:**
```bash
tail -f /root/online_rl/conrft/examples/experiments/task1_pick_banana/conrft/learner.log
```

**Actor logs:**
```bash
tail -f /root/online_rl/conrft/examples/experiments/task1_pick_banana/conrft/actor.log
```

### Check Network Traffic

**On robot machine:**
```bash
# Monitor Pyro5 traffic
sudo tcpdump -i any port 9090 -n

# Monitor franka server traffic
sudo tcpdump -i any port 5000 -n
```

**On A800:**
```bash
# Monitor agentlace traffic
sudo tcpdump -i any port 3333 or port 3334 -n
```

## Summary

✅ **Robot Machine**: Run `franka_server.py` + `env_server.py`
✅ **A800 Machine**: Run `learner` + `actor` (no hardware dependencies)
✅ **Network**: Configure firewall to allow ports 5000, 9090, 3333, 3334
✅ **Test**: Use `test_pyro_env.py` to verify connection before training

You're ready to start distributed training! 🚀
