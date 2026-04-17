# Pyro5 Remote Environment Deployment Guide

## Architecture Overview

```
Local Machine (Franka + RealSense)          A800 Machine (GPU Server)
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

## Key Advantages

✅ **A800 完全不需要安装硬件依赖**
- 不需要 RealSense SDK (`pyrealsense2`)
- 不需要 ROS (`rosbag`, `rospy`)
- 不需要 `franka_env` 包
- 只需要 `serl_launcher` 和 `Pyro5`

✅ **最小改动**
- 只添加了 3 个 flag 参数
- 只修改了环境创建逻辑（10 行代码）
- actor 循环代码完全不变

✅ **网络传输优化**
- Pyro5 自动序列化 numpy 数组
- 只传输 obs/reward/done 等小数据
- 图像数据已经在本地处理

## Deployment Steps

### Step 1: Local Machine Setup

#### 1.1 Install Pyro5
```bash
pip install Pyro5
```

#### 1.2 Configure franka_server.py
修改 `serl_robot_infra/robot_servers/franka_server.py`:
```python
# Line 15: Change from "127.0.0.1" to "0.0.0.0"
parser.add_argument("--flask_url", type=str, default="0.0.0.0")
```

#### 1.3 Configure config.py
修改 `examples/experiments/task1_pick_banana/config.py`:
```python
# Line 15: Change to local machine IP
SERVER_URL = "http://<LOCAL_MACHINE_IP>:5000/"
```

#### 1.4 Start Servers
```bash
# Terminal 1: Start franka server
cd serl_robot_infra/robot_servers
python franka_server.py --flask_url 0.0.0.0

# Terminal 2: Start Pyro5 environment server
cd serl_robot_infra
python env_server.py --host 0.0.0.0 --port 9090 --task task1_pick_banana
```

### Step 2: A800 Machine Setup

#### 2.1 Install Pyro5
```bash
pip install Pyro5
```

#### 2.2 Install serl_launcher (without hardware dependencies)
```bash
cd serl_launcher
pip install -e .
```

**注意**: A800 不需要安装 `serl_robot_infra`！

#### 2.3 Test Connection
```bash
cd serl_launcher
python test_pyro_env.py --ip <LOCAL_MACHINE_IP> --port 9090
```

Expected output:
```
Testing Pyro5 Environment Connection
============================================================

Connecting to remote environment at <LOCAL_MACHINE_IP>:9090...
Connected to remote environment at PYRO:remote_env@<LOCAL_MACHINE_IP>:9090
Remote action space: Box(-1.0, 1.0, (7,), float32)
Remote observation space: Dict(...)

[Test 1] Action Space:
  Shape: (7,)
  Low: [-1. -1. -1. -1. -1. -1. -1.]
  High: [1. 1. 1. 1. 1. 1. 1.]

[Test 2] Observation Space:
  state: shape=(15,), dtype=float32
  images: shape=(2, 84, 84), dtype=uint8
  ...

All tests passed!
```

### Step 3: Start Training

#### 3.1 Start Learner (on A800)
```bash
cd examples
python train_conrft_octo.py \
    --exp_name task1_pick_banana \
    --learner \
    --ip <A800_IP> \
    --demo_path /path/to/demo.pkl \
    --checkpoint_path /path/to/checkpoints
```

#### 3.2 Start Actor (on A800)
```bash
cd examples
python train_conrft_octo.py \
    --exp_name task1_pick_banana \
    --actor \
    --ip <A800_IP> \
    --use_pyro_env \
    --pyro_env_ip <LOCAL_MACHINE_IP> \
    --pyro_env_port 9090
```

## Configuration Details

### Pyro5 Serialization

Pyro5 默认使用 `serpent` 序列化器，但对于 numpy 数组，`pickle` 更高效。我们在代码中配置了：

```python
import Pyro5.config
Pyro5.config.SERIALIZER = "pickle"
Pyro5.config.SERIALIZERS_ACCEPTED = {"pickle", "serpent"}
```

### Network Ports

| Port | Service | Direction | Description |
|------|---------|-----------|-------------|
| 5000 | franka_server.py | Local → Local | Flask HTTP for robot control |
| 9090 | env_server.py | Local → A800 | Pyro5 remote env calls |
| 3333 | agentlace | A800 ↔ A800 | TrainerServer (learner) |
| 3334 | agentlace | A800 ↔ A800 | Broadcast (learner → actor) |

### Firewall Configuration

**Local Machine**:
```bash
# Allow A800 to access Pyro5 server
sudo ufw allow from <A800_IP> to any port 9090
sudo ufw allow from <A800_IP> to any port 5000
```

**A800 Machine**:
```bash
# Allow actor to connect to learner
sudo ufw allow from <A800_IP> to any port 3333
sudo ufw allow from <A800_IP> to any port 3334
```

## Troubleshooting

### Issue 1: Connection Refused
```
Pyro5.errors.ConnectionRefusedError: Cannot connect to remote_env@<IP>:9090
```

**Solution**:
1. Check if `env_server.py` is running on local machine
2. Check firewall rules
3. Verify IP address is correct

### Issue 2: Serialization Error
```
Pyro5.errors.SerializeError: Cannot serialize object of type <class '...'>
```

**Solution**:
1. Ensure Pyro5 config is set to use pickle
2. Check if info dict contains non-serializable objects (e.g., ROS messages)
3. Add custom serialization if needed

### Issue 3: High Latency
```
Warning: Step took >100ms
```

**Solution**:
1. Check network bandwidth between machines
2. Reduce image resolution in config
3. Use compression for large observations

### Issue 4: Import Error on A800
```
ModuleNotFoundError: No module named 'franka_env'
```

**Solution**:
This is expected! A800 should NOT have `franka_env` installed. The Pyro5 wrapper avoids this dependency.

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

## Alternative: ZeroRPC

如果 Pyro5 有问题，可以考虑 ZeroRPC:

```python
# env_server.py (ZeroRPC version)
import zerorpc

class RemoteEnv:
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return [obs, reward, done, truncated, info]

s = zerorpc.Server(RemoteEnv())
s.bind("tcp://0.0.0.0:9090")
s.run()
```

```python
# pyro_env_wrapper.py (ZeroRPC version)
import zerorpc

class ZeroRPCEnvWrapper:
    def __init__(self, ip, port):
        self.client = zerorpc.Client(f"tcp://{ip}:{port}")
        # ... similar to Pyro5 version
```

ZeroRPC 使用 MessagePack 序列化，对 numpy 数组支持也很好。

## Summary

这个 Pyro5 方案完全可行，而且改动最小：

1. ✅ A800 不需要安装任何硬件依赖
2. ✅ 只修改了环境创建逻辑（10 行代码）
3. ✅ Actor 循环代码完全不变
4. ✅ 网络传输开销可控（~30-70ms）
5. ✅ 架构清晰，易于调试

你可以直接按照上面的步骤部署了！
