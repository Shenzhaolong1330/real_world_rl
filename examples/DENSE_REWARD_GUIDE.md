# 稠密奖励训练流程指南

本指南说明如何在 real_world_rl 中使用稠密奖励进行机器人学习。

## 两种稠密奖励方案

### 方案 1: 分类器稠密奖励（简单）
- 使用训练好的二分类器判断成功/失败
- 需要人工标注数据
- 适合简单任务

### 方案 2: GRM 稠密奖励（推荐）
- 使用 Robo-Dopamine 的 GRM 模型
- 预训练模型，泛化能力强
- 基于视频序列推理奖励
- 适合复杂任务

## 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                         云端服务器                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ record_demos_   │  │ train_reward_   │  │ train_conrft_   │ │
│  │ octo.py         │  │ classifier.py   │  │ octo.py         │ │
│  │ (数据收集)       │  │ (训练分类器)     │  │ (训练Learner)   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│           │                                                      │
│           │ Pyro5 远程调用                                        │
│           │                                                      │
└───────────┼──────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      本地机器人电脑                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   server.py     │  │  机器人控制程序  │  │    真实环境      │ │
│  │ (Pyro5服务)     │  │  (Franka/UR)    │  │   (FrankaEnv)   │ │
│  │  端口: 9090     │  │                 │  │                 │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│           ▲                                                      │
│           │ 直接调用                                              │
│  ┌────────┴────────┐                                             │
│  │record_success_ │                                             │
│  │    fail.py     │  ← 本地数据收集                              │
│  └─────────────────┘                                             │
└─────────────────────────────────────────────────────────────────┘
```

## 完整流程

### 方案 1: 分类器稠密奖励流程

#### 步骤 1: 收集成功/失败数据（本地机器人电脑）

用于训练奖励分类器：

```bash
# 在本地机器人电脑上运行
cd ~/real_world_rl/examples
python record_success_fail.py --exp_name=task2_insert_vial --successes_needed=200

# 操作说明：
# - 使用 spacemouse 遥操作机器人
# - 当达到成功状态时，快速双击 '.' 键标记成功
# - 收集足够样本后自动保存
```

**输出：**
- `classifier_data/task2_insert_vial_200_success_images_<timestamp>.pkl`
- `classifier_data/task2_insert_vial_failure_images_<timestamp>.pkl`

### 步骤 2: 训练奖励分类器（云端服务器）
```bash
# 在云端服务器运行
cd ~/real_world_rl/examples
python train_reward_classifier.py --exp_name=task2_insert_vial --num_epochs=1500
```

**输出：**
- `classifier_ckpt/classifier_1500` - 训练好的分类器模型

### 方案 2: GRM 稠密奖励流程（推荐）

#### 步骤 1: 安装 Robo-Dopamine

```bash
# 安装 robo-dopamine
pip install robo-dopamine

# 或者从源码安装
git clone https://github.com/tanhuajie2001/Robo-Dopamine.git
cd Robo-Dopamine
pip install -e .
```

#### 步骤 2: 下载 GRM 模型权重

```bash
# 下载预训练模型
mkdir -p ~/real_world_rl/weights
cd ~/real_world_rl/weights

# 从 HuggingFace 下载
# 方式 1: 使用 huggingface-cli
huggingface-cli download tanhuajie2001/Robo-Dopamine-GRM-2.0-8B-Preview --local-dir ./Robo-Dopamine-GRM-2.0-8B-Preview

# 方式 2: 手动下载
# 访问 https://huggingface.co/tanhuajie2001/Robo-Dopamine-GRM-2.0-8B-Preview
```

#### 步骤 3: 配置 GRM 路径

在 `experiments/task2_insert_vial/config.py` 中配置：

```python
class TrainConfig(DefaultTrainingConfig):
    # GRM 稠密奖励配置
    use_grm_reward = True  # 启用 GRM 稠密奖励
    weight_path = "/home/szl/real_world_rl/weights/Robo-Dopamine-GRM-2.0-8B-Preview"
    frame_interval = 4  # GRM 推理的帧间隔
    batch_dopamine = 45  # GRM 批处理大小
```

#### 步骤 4: 收集演示数据（使用 GRM 稠密奖励）

```bash
# 本地模式
cd ~/real_world_rl/examples
python record_demos_octo_dopamine.py \
    --exp_name=task2_insert_vial \
    --successes_needed=30

# 远程模式
python record_demos_octo_dopamine.py \
    --exp_name=task2_insert_vial \
    --use_pyro_env \
    --pyro_env_ip=<本地机器人IP> \
    --successes_needed=30
```

**输出：**
- `demo_data/task2_insert_vial_30_demos_dopamine_<timestamp>.pkl` - 带 GRM 稠密奖励的演示数据

### 通用步骤: 训练策略（云端服务器）
```bash
# 在云端服务器运行
cd ~/real_world_rl/examples
python train_reward_classifier.py --exp_name=task2_insert_vial --num_epochs=1500
```

**输出：**
- `classifier_ckpt/classifier_1500` - 训练好的分类器模型

### 步骤 3: 收集演示数据（云端或本地）

#### 方式 A: 云端远程收集（推荐）

```bash
# 1. 在本地机器人电脑启动 server.py
python server.py

# 2. 在云端服务器运行
cd ~/real_world_rl/examples
python record_demos_octo.py \
    --exp_name=task2_insert_vial \
    --use_pyro_env \
    --pyro_env_ip=<本地机器人IP> \
    --pyro_env_port=9090 \
    --successes_needed=30
```

#### 方式 B: 本地直接收集

```bash
# 在本地机器人电脑上运行
cd ~/real_world_rl/examples
python record_demos_octo.py \
    --exp_name=task2_insert_vial \
    --successes_needed=30
```

**输出：**
- `demo_data/task2_insert_vial_30_demos_<timestamp>.pkl`

### 步骤 4: 训练策略（云端服务器）

#### Actor 模式（本地机器人电脑）

```bash
# 在本地机器人电脑运行
cd ~/real_world_rl/examples
python train_conrft_octo.py \
    --exp_name=task2_insert_vial \
    --actor \
    --ip=<云端服务器IP> \
    --use_pyro_env \
    --pyro_env_ip=<本地机器人IP>
```

#### Learner 模式（云端服务器）

```bash
# 在云端服务器运行
cd ~/real_world_rl/examples
python train_conrft_octo.py \
    --exp_name=task2_insert_vial \
    --learner \
    --demo_path=./demo_data/task2_insert_vial_30_demos_<timestamp>.pkl
```

## 关键组件说明

### 1. CompletionRewardClassifierWrapper

用于人工标记成功状态的 wrapper：

```python
from franka_env.envs.wrappers import CompletionRewardClassifierWrapper

env = CompletionRewardClassifierWrapper(env)
# 双击 '.' 键标记成功
```

### 2. MultiCameraBinaryRewardClassifierWrapper

使用训练好的分类器计算奖励：

```python
from franka_env.envs.wrappers import MultiCameraBinaryRewardClassifierWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

classifier = load_classifier_func(
    key=jax.random.PRNGKey(0),
    sample=env.observation_space.sample(),
    image_keys=["side_classifier"],
    checkpoint_path="classifier_ckpt/",
)

def reward_func(obs):
    sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
    if sigmoid(classifier(obs)[0]) > 0.9:
        return 10.0  # 成功奖励
    else:
        return -0.05  # 失败惩罚

env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
```

### 3. Pyro5 远程环境

支持云端-本地分离部署：

```python
# 本地运行 server.py
python server.py  # 监听 0.0.0.0:9090

# 云端连接
from serl_launcher.utils.pyro_env_wrapper import create_pyro_env
env = create_pyro_env("本地IP", 9090)
```

## 配置文件说明

每个任务需要在 `experiments/<task_name>/config.py` 中配置：

```python
class TrainConfig(DefaultTrainingConfig):
    image_keys = ["side_policy_256", "wrist_1"]  # 策略使用的相机
    classifier_keys = ["side_classifier"]  # 分类器使用的相机
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    
    reward_neg = -0.05  # 失败惩罚
    task_desc = "Insert the vial into the rack"  # 任务描述（用于 Octo）
    
    def get_environment(self, fake_env=False, save_video=False, classifier=False, stack_obs_num=1):
        # 创建环境并添加 wrappers
        env = ...
        if classifier:
            # 添加奖励分类器 wrapper
            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        return env
```

## 常见问题

### Q1: 摄像头设备被占用

```bash
# 查找占用进程
sudo lsof /dev/video*

# 结束进程
sudo kill -9 <PID>

# 重置摄像头驱动
sudo modprobe -r uvcvideo && sudo modprobe uvcvideo
```

### Q2: Pyro5 连接失败

```bash
# 检查本地 server.py 是否运行
ps aux | grep server.py

# 检查端口是否开放
netstat -tulpn | grep 9090

# 检查防火墙
sudo ufw allow 9090
```

### Q3: JAX 版本兼容性问题

```bash
# 检查 JAX 版本
python -c "import jax; print(jax.__version__)"

# 如果遇到 KeyArray 错误，修改 octo/utils/typing.py:
# PRNGKey = jax.Array  # 新版本
```

## 文件结构

```
real_world_rl/examples/
├── record_success_fail.py          # 收集成功/失败数据
├── train_reward_classifier.py      # 训练奖励分类器
├── record_demos_octo.py            # 收集演示数据
├── train_conrft_octo.py            # 训练策略
├── server.py                        # Pyro5 服务器（本地）
├── classifier_data/                # 分类器训练数据
├── classifier_ckpt/                # 分类器模型
├── demo_data/                       # 演示数据
└── experiments/
    ├── mappings.py                  # 任务映射
    ├── config.py                    # 基础配置
    └── task2_insert_vial/
        ├── config.py                # 任务配置
        └── wrapper.py               # 任务环境 wrapper
```

## 参考资料

- [Pyro5 文档](https://pyro5.readthedocs.io/)
- [JAX 文档](https://jax.readthedocs.io/)
- [Octo 模型](https://octo-models.github.io/)
- [SERL Launcher](https://github.com/rail-berkeley/serl_launcher)
