# Task1 部署说明（A800 + 本地机器人）

这份文档只保留你现在实际需要的流程：

- A800 上分开启动 `learner` 和 `actor`
- 本地机器人机分开启动 `env server` 和 `serl_robot_infra/robot_servers/launch_right_server.sh`

> 默认你使用环境 `conrft`。每个终端都先激活该环境。

---

## 1. 角色与进程

### 本地机器人机（Franka + RealSense）

1. `serl_robot_infra/robot_servers/launch_right_server.sh`
  - 启动 ROS/Franka 控制与 Flask 机器人服务（默认端口 `5000`）
2. `examples/experiments/task1_pick_banana/run_env_server.sh`
  - 启动 Pyro5 环境服务（默认端口 `9090`）

### A800 训练机

1. `examples/experiments/task1_pick_banana/run_learner_conrft_remote.sh`
  - 启动 learner（agentlace 监听）
2. `examples/experiments/task1_pick_banana/run_actor_conrft_remote.sh`
  - 启动 actor（通过 Pyro5 连接本地 env server）

---

## 2. 启动前检查（一次性）

### 2.1 本地机器人机

1. 确认 `franka_server` 对外监听：
  - `launch_right_server.sh` 中 `--flask_url="0.0.0.0"`
2. 确认 `env server` 对外监听：
  - `run_env_server.sh` 中 `LOCAL_IP="0.0.0.0"`
3. 确认任务配置访问机器人服务地址：
  - `examples/experiments/task1_pick_banana/config.py` 的 `SERVER_URL`
  - 同机部署可用 `http://127.0.0.2:5000/`（你当前配置）

### 2.2 A800

1. 修改 `examples/experiments/task1_pick_banana/run_actor_conrft_remote.sh`：
  - `ROBOT_IP="<本地机器人机IP>"`
2. learner 与 actor 的 `CHECKPOINT_PATH` 保持一致。

---

## 3. 启动顺序（严格按顺序）

> 推荐 4 个终端。

### 终端 L1（本地机器人机）启动机器人控制服务

```bash
cd /home/deepcybo/worksplace/real_world_rl
conda activate conrft
bash serl_robot_infra/robot_servers/launch_right_server.sh
```

### 终端 L2（本地机器人机）启动 Pyro5 环境服务

```bash
cd /home/deepcybo/worksplace/real_world_rl
conda activate conrft
bash examples/experiments/task1_pick_banana/run_env_server.sh
```

看到类似输出表示成功：

```text
Pyro5 Environment Server Started
URI: PYRO:remote_env@0.0.0.0:9090
On A800 machine, use this URI to connect:
  PYRO:remote_env@<LOCAL_MACHINE_IP>:9090
```

### 终端 A1（A800）启动 learner

```bash
cd /home/deepcybo/worksplace/real_world_rl
conda activate conrft
bash examples/experiments/task1_pick_banana/run_learner_conrft_remote.sh
```

### 终端 A2（A800）启动 actor

```bash
cd /home/deepcybo/worksplace/real_world_rl
conda activate conrft
bash examples/experiments/task1_pick_banana/run_actor_conrft_remote.sh
```

---

## 4. 快速连通性检查

在 A800 上运行：

```bash
cd /home/deepcybo/worksplace/real_world_rl
conda activate conrft
python serl_launcher/test_pyro_env.py --ip <本地机器人机IP> --port 9090
```

如果通过，说明 A800 能连到本地 `env server`。

---

## 5. 常见问题

### 5.1 actor 连不上 env server

- 检查 `run_actor_conrft_remote.sh` 里的 `ROBOT_IP`
- 检查本地 `run_env_server.sh` 是否在运行
- 检查端口：`9090`（Pyro5）和 `5000`（Flask）

### 5.2 SpaceMouse / 相机问题

- SpaceMouse 正常应看到 `SpaceMouse Wireless BT found`
- 相机超时 `Frame didn't arrive within 5000`：
  - 优先用 USB 直连，不要挂廉价 HUB
  - 降低相机分辨率或帧率（在 `config.py`）

### 5.3 路径不一致

你仓库里有部分脚本写了历史路径（如 `/root/hil-serl/...`、`/home/szl/...`）。
若启动报 `No such file or directory`，请按当前机器路径改脚本中的 `cd`。

---

## 6. 推荐最小工作流

1. 本地先起 `launch_right_server.sh`
2. 本地再起 `run_env_server.sh`
3. A800 起 `run_learner_conrft_remote.sh`
4. A800 起 `run_actor_conrft_remote.sh`

只要这四步都在运行，你的远程训练链路就是完整的（robot/env ↔ actor ↔ learner）。
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
