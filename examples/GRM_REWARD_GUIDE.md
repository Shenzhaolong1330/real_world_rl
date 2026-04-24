# GRM 稠密奖励使用指南

## 概述

GRM (Generalist Reward Model) 是 Robo-Dopamine 项目提供的预训练奖励模型，能够根据视频序列推理出密集的奖励信号。

## 优势

相比传统的分类器稠密奖励，GRM 具有以下优势：

1. **泛化能力强**: 预训练在大规模机器人数据上，能适应多种任务
2. **无需标注**: 不需要人工标注成功/失败数据
3. **密集奖励**: 提供平滑的奖励曲线，有利于策略学习
4. **多模态理解**: 基于视觉-语言模型，能理解任务描述

## 安装

### 1. 安装 Robo-Dopamine

```bash
# 方式 1: 从 PyPI 安装
pip install robo-dopamine

# 方式 2: 从源码安装
git clone https://github.com/tanhuajie2001/Robo-Dopamine.git
cd Robo-Dopamine
pip install -e .
```

### 2. 下载 GRM 模型权重

```bash
# 创建权重目录
mkdir -p ~/real_world_rl/weights
cd ~/real_world_rl/weights

# 从 HuggingFace 下载
huggingface-cli download tanhuajie2001/Robo-Dopamine-GRM-2.0-8B-Preview \
    --local-dir ./Robo-Dopamine-GRM-2.0-8B-Preview

# 或者手动下载
# 访问: https://huggingface.co/tanhuajie2001/Robo-Dopamine-GRM-2.0-8B-Preview
```

## 配置

### 1. 更新任务配置

在 `experiments/<task_name>/config.py` 中添加 GRM 配置：

```python
class TrainConfig(DefaultTrainingConfig):
    # ... 其他配置 ...
    
    # GRM 稠密奖励配置
    use_grm_reward = True  # 启用 GRM 稠密奖励
    weight_path = "/home/szl/real_world_rl/weights/Robo-Dopamine-GRM-2.0-8B-Preview"
    frame_interval = 4  # GRM 推理的帧间隔
    batch_dopamine = 45  # GRM 批处理大小
    visualize = False  # 是否可视化奖励
    only_vis_avg = False  # 是否只可视化平均奖励
```

### 2. 配置相机

GRM 需要多个相机视角的视频序列。确保环境配置中包含以下相机：

```python
class EnvConfig(DefaultEnvConfig):
    REALSENSE_CAMERAS = {
        "side_policy_256": {  # 高位相机
            "serial_number": "...",
            "dim": (1280, 720),
            "exposure": 13000,
        },
        "wrist_1": {  # 手腕相机
            "serial_number": "...",
            "dim": (1280, 720),
            "exposure": 10500,
        },
        # 可选: 第三个相机
    }
```

## 使用流程

### 方式 1: 使用 record_demos_octo_dopamine.py

专门用于收集带 GRM 稠密奖励的演示数据：

```bash
# 本地模式
cd ~/real_world_rl/examples
python record_demos_octo_dopamine.py \
    --exp_name=task2_insert_vial \
    --successes_needed=30

# 远程模式（云端控制本地机器人）
python record_demos_octo_dopamine.py \
    --exp_name=task2_insert_vial \
    --use_pyro_env \
    --pyro_env_ip=<本地机器人IP> \
    --successes_needed=30
```

**输出：**
- `demo_data/task2_insert_vial_30_demos_dopamine_<timestamp>.pkl`

### 方式 2: 在训练中使用 GRM

修改 `train_conrft_octo.py` 的 actor 部分：

```python
# 在 actor 函数开头加载 GRM 模型
if config.use_grm_reward:
    from robo_dopamine.examples.inference import GRMInference
    grm_model = GRMInference(config.weight_path, gpu_memory_utilization=0.5)
    
# 在轨迹处理时使用 GRM 推理奖励
if done and config.use_grm_reward:
    # 提取相机帧
    trajectory, high_frames, left_frames, right_frames = process_trajectory(trajectory)
    
    # GRM 推理
    avg_reward = inference_reward(
        model=grm_model,
        cam_high_frames=high_frames,
        cam_left_frames=left_frames,
        cam_right_frames=right_frames,
        ...
    )
    
    # 更新奖励
    for i, transition in enumerate(trajectory):
        transition["rewards"] = avg_reward[i]
```

## GRM 推理参数说明

### RewardConfig

```python
@dataclass
class RewardConfig:
    gamma: float  # 折扣因子
    task_instruction: str  # 任务描述
    output_root: Path  # 输出目录
    goal_image_path: Path  # 目标图像路径
    frame_interval: int  # 帧间隔（默认 4）
    batch_size: int  # 批处理大小（默认 45）
    visualize: bool  # 是否可视化
    only_vis_avg: bool  # 是否只可视化平均奖励
```

### inference_reward 参数

- `model`: GRMInference 实例
- `cam_high_frames`: 高位相机帧列表
- `cam_left_frames`: 左侧相机帧列表
- `cam_right_frames`: 右侧相机帧列表
- `task`: 任务描述字符串
- `frame_interval`: 每隔多少帧推理一次
- `batch_size`: 批处理大小
- `gamma`: 折扣因子
- `completion_reward`: 完成奖励列表

## 性能优化

### 1. GPU 内存管理

```python
# 调整 GPU 内存使用率
grm_model = GRMInference(
    config.weight_path,
    gpu_memory_utilization=0.5  # 降低到 50%
)
```

### 2. 批处理优化

```python
# 增加批处理大小以提高吞吐量
batch_dopamine = 60  # 默认 45
```

### 3. 帧间隔调整

```python
# 增加帧间隔以减少推理次数
frame_interval = 8  # 默认 4
```

## 常见问题

### Q1: GPU 内存不足

```bash
# 降低 GPU 内存使用率
grm_model = GRMInference(path, gpu_memory_utilization=0.3)

# 或使用更小的模型
# Robo-Dopamine-GRM-2.0-4B-Preview
```

### Q2: 推理速度慢

```python
# 增加帧间隔
frame_interval = 8

# 增加批处理大小
batch_dopamine = 60

# 关闭可视化
visualize = False
```

### Q3: 奖励值异常

```python
# 检查任务描述是否清晰
task_desc = "Insert the vial into the rack"  # 清晰的任务描述

# 检查相机配置
# 确保相机视角能清晰看到任务执行过程
```

### Q4: 模型下载失败

```bash
# 使用镜像站
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载后指定路径
weight_path = "/path/to/downloaded/model"
```

## 对比：分类器 vs GRM

| 特性 | 分类器稠密奖励 | GRM 稠密奖励 |
|------|---------------|-------------|
| 需要标注数据 | ✅ 是 | ❌ 否 |
| 泛化能力 | ⚠️ 有限 | ✅ 强 |
| 计算开销 | ✅ 低 | ⚠️ 高 |
| 实时性 | ✅ 实时 | ⚠️ 需要批处理 |
| 适用场景 | 简单任务 | 复杂任务 |
| 部署难度 | ✅ 简单 | ⚠️ 需要大模型 |

## 最佳实践

1. **简单任务**: 使用分类器稠密奖励
2. **复杂任务**: 使用 GRM 稠密奖励
3. **资源受限**: 使用分类器或更小的 GRM 模型
4. **实时性要求高**: 使用分类器
5. **泛化性要求高**: 使用 GRM

## 示例项目

查看 `record_demos_octo_dopamine.py` 获取完整示例。

## 参考资料

- [Robo-Dopamine GitHub](https://github.com/tanhuajie2001/Robo-Dopamine)
- [GRM Paper](https://arxiv.org/abs/...)
- [HuggingFace Model](https://huggingface.co/tanhuajie2001/Robo-Dopamine-GRM-2.0-8B-Preview)
