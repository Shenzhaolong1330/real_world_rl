"""
使用 Octo + GRM/Dopamine 的演示数据收集脚本

功能说明：
- 收集人类遥操作的演示数据
- 通过 Pyro5 远程连接机器人环境
- 使用 Octo 模型计算动作嵌入
- 使用 GRM (Generalist Reward Model) 计算密集奖励
- 仅保存成功的演示轨迹

特点：
- 结合了 Octo 的语言条件嵌入和 GRM 的奖励推理
- 生成带有密集奖励信号的演示数据
- 适用于 Robo-Dopamine 训练框架

使用方式：
  # 本地模式
  python record_demos_octo_dopamine.py --exp_name=task2_insert_vial --successes_needed=30
  
  # 远程模式
  python record_demos_octo_dopamine.py --exp_name=task2_insert_vial --use_pyro_env --pyro_env_ip=<机器人IP> --successes_needed=30
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
import copy
import pickle as pkl
import cv2
from pathlib import Path
from datetime import datetime
from copy import deepcopy
from absl import app, flags
import time

from experiments.mappings import CONFIG_MAPPING
from data_util import add_mc_returns_to_trajectory, add_embeddings_to_trajectory, add_next_embeddings_to_trajectory

from octo.model.octo_model import OctoModel

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "task2_insert_vial", "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 30, "Number of successful demos to collect.")
flags.DEFINE_float("reward_scale", 1.0, "reward_scale ")
flags.DEFINE_float("reward_bias", 0.0, "reward_bias")
flags.DEFINE_boolean("use_pyro_env", False, "Use Pyro5 remote environment")
flags.DEFINE_string("pyro_env_ip", None, "IP address of the Pyro5 environment server")
flags.DEFINE_integer("pyro_env_port", 9090, "Port of the Pyro5 environment server")
flags.DEFINE_string(
    "octo_path",
    "/home/szl/real_world_rl/octo-small",
    "Octo checkpoint path.",
)
flags.DEFINE_string(
    "grm_path",
    None,
    "GRM model path (optional, defaults to config.weight_path)",
)


@dataclass
class RewardConfig:
    """Configuration for GRM reward inference"""
    gamma: float
    task_instruction: str
    output_root: Path
    goal_image_path: Path
    frame_interval: int
    batch_size: int
    visualize: bool
    only_vis_avg: bool
    
    @property
    def run_roots(self):
        return {"high": self.output_root / "cam_high", 
                "left": self.output_root / "cam_left",
                "right": self.output_root / "cam_right"}
    
    @property
    def cam_dirs_map(self):
        return {
            "high": str(self.run_roots["high"]),
            "left": str(self.run_roots["left"]),
            "right": str(self.run_roots["right"])
        }
    
    @property
    def ref_end_map(self):
        return {
            "high": {"ref": str(self.goal_image_path), "end": str(self.run_roots["high"] / "end.jpg")},
            "left": {"ref": str(self.goal_image_path), "end": str(self.run_roots["left"] / "end.jpg")},
            "right": {"ref": str(self.goal_image_path), "end": str(self.run_roots["right"] / "end.jpg")}
        }


def inference_reward(model, cam_high_frames, cam_left_frames, cam_right_frames,
                      run_roots, cam_dirs_map, ref_end_map, task, frame_interval,
                      batch_size, visualize, only_vis_avg, gamma, completion_reward):
    """
    使用 GRM 模型推理密集奖励
    
    Args:
        model: GRMInference 模型实例
        cam_high_frames: 高位相机帧列表
        cam_left_frames: 左侧相机帧列表
        cam_right_frames: 右侧相机帧列表
        ...
    
    Returns:
        avg_reward: 平均奖励列表
    """
    try:
        from robo_dopamine.examples.inference import build_samples_json, make_sample_indices_by_interval
    except ImportError:
        print("⚠ robo_dopamine not installed. Install with:")
        print("  pip install robo-dopamine")
        raise
    
    # 创建输出目录
    for cam_name, root in run_roots.items():
        os.makedirs(root, exist_ok=True)
    
    # 保存帧图像
    for i, (high_frame, left_frame, right_frame) in enumerate(zip(cam_high_frames, cam_left_frames, cam_right_frames)):
        cv2.imwrite(str(run_roots["high"] / f"{i:04d}.jpg"), high_frame)
        cv2.imwrite(str(run_roots["left"] / f"{i:04d}.jpg"), left_frame)
        cv2.imwrite(str(run_roots["right"] / f"{i:04d}.jpg"), right_frame)
    
    # 保存最后一帧作为 end.jpg
    if len(cam_high_frames) > 0:
        cv2.imwrite(str(run_roots["high"] / "end.jpg"), cam_high_frames[-1])
        cv2.imwrite(str(run_roots["left"] / "end.jpg"), cam_left_frames[-1])
        cv2.imwrite(str(run_roots["right"] / "end.jpg"), cam_right_frames[-1])
    
    # 构建样本 JSON
    samples = build_samples_json(
        cam_dirs_map=cam_dirs_map,
        ref_end_map=ref_end_map,
        task=task,
        frame_interval=frame_interval,
        batch_size=batch_size,
    )
    
    # 推理奖励
    results = model.inference_batch(samples)
    
    # 解析结果
    rewards = []
    for result in results:
        pred = result.get("pred", "")
        # 解析预测的奖励值
        try:
            reward = float(pred.split("Score: ")[1].split("/")[0])
        except:
            reward = 0.0
        rewards.append(reward)
    
    # 插值到完整轨迹长度
    sample_indices = make_sample_indices_by_interval(len(cam_high_frames), frame_interval)
    avg_reward = np.interp(np.arange(len(cam_high_frames)), sample_indices, rewards)
    
    # 归一化奖励
    if len(completion_reward) > 0 and completion_reward[-1] > 0:
        # 如果任务成功，缩放奖励
        avg_reward = avg_reward * (1.0 / (avg_reward.max() + 1e-6))
    
    return avg_reward


def process_trajectory(trajectory):
    """处理轨迹，提取相机帧"""
    high_frame_list = []
    left_frame_list = []
    right_frame_list = []
    
    for t in trajectory:
        obs = t["observations"]
        
        # 提取图像帧（根据实际相机配置调整）
        if "side_policy_256" in obs:
            # side_policy_256 作为 high camera
            high_frame = obs["side_policy_256"]
            if high_frame.ndim == 4:
                high_frame = high_frame[0]
            high_frame = (high_frame * 255).astype(np.uint8)
            high_frame_list.append(cv2.cvtColor(high_frame, cv2.COLOR_RGB2BGR))
        
        if "wrist_1" in obs:
            # wrist_1 作为 left camera
            left_frame = obs["wrist_1"]
            if left_frame.ndim == 4:
                left_frame = left_frame[0]
            left_frame = (left_frame * 255).astype(np.uint8)
            left_frame_list.append(cv2.cvtColor(left_frame, cv2.COLOR_RGB2BGR))
        
        # 如果有第三个相机
        right_frame_list.append(np.zeros((128, 128, 3), dtype=np.uint8))
    
    return trajectory, high_frame_list, left_frame_list, right_frame_list


def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    
    # 设置路径
    octo_path = FLAGS.octo_path or config.octo_path
    grm_path = FLAGS.grm_path or getattr(config, 'weight_path', None)
    
    if grm_path is None:
        print("⚠ GRM path not configured. Add 'weight_path' to config or use --grm_path")
        print("  Example: --grm_path=/path/to/Robo-Dopamine-GRM-2.0-8B-Preview")
        return
    
    # 创建环境
    if FLAGS.use_pyro_env:
        from serl_launcher.utils.pyro_env_wrapper import create_pyro_env
        print(f"Connecting to Pyro5 remote environment at {FLAGS.pyro_env_ip}:{FLAGS.pyro_env_port}")
        env = create_pyro_env(FLAGS.pyro_env_ip, FLAGS.pyro_env_port)
    else:
        env = config.get_environment(fake_env=False, save_video=False, classifier=False, stack_obs_num=2)
    
    # 加载 Octo 模型
    print(f"Loading Octo model from: {octo_path}")
    octo_model = OctoModel.load_pretrained(octo_path)
    tasks = octo_model.create_tasks(texts=[config.task_desc])
    
    # 加载 GRM 模型
    print(f"Loading GRM model from: {grm_path}")
    try:
        from robo_dopamine.examples.inference import GRMInference
        grm_model = GRMInference(grm_path, gpu_memory_utilization=0.5)
        print("✓ GRM model loaded successfully")
    except ImportError:
        print("✗ robo_dopamine not installed. Install with:")
        print("  pip install robo-dopamine")
        return
    
    # 设置输出目录
    output_root = Path(__file__).resolve().parent / "experiments" / FLAGS.exp_name / "results"
    os.makedirs(output_root, exist_ok=True)
    goal_image_path = Path(__file__).resolve().parent / "experiments" / FLAGS.exp_name / "goal_image.png"
    
    obs, info = env.reset()
    print(obs.keys())
    print("Reset done")
    
    transitions = []
    success_count = 0
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    trajectory = []
    completion_rewards = []
    returns = 0
    
    print("\n" + "="*60)
    print("操作说明:")
    print("  - 使用 spacemouse 遥操作机器人执行任务")
    print("  - GRM 会自动计算密集奖励")
    print("  - 按 Ctrl+C 可提前退出")
    print("="*60 + "\n")
    
    try:
        while success_count < success_needed:
            actions = np.zeros(env.action_space.sample().shape)
            next_obs, rew, done, truncated, info = env.step(actions)
            returns += rew
            completion_rewards.append(rew)
            
            if "intervene_action" in info:
                actions = info["intervene_action"]
            
            transition = deepcopy(
                dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=rew,
                    masks=1.0 - done,
                    dones=done,
                    infos=info,
                )
            )
            obs = next_obs
            trajectory.append(transition)
            
            pbar.set_description(f"Return: {returns:.2f}")
            print(f"FPS: N/A, Reward: {rew:.4f}")
            
            if done:
                # 处理轨迹
                trajectory, high_frame_list, left_frame_list, right_frame_list = process_trajectory(trajectory)
                
                # 使用 GRM 推理密集奖励
                rew_cfg = RewardConfig(
                    gamma=config.discount,
                    task_instruction=config.task_desc,
                    output_root=output_root,
                    goal_image_path=goal_image_path,
                    frame_interval=getattr(config, 'frame_interval', 4),
                    batch_size=getattr(config, 'batch_dopamine', 45),
                    visualize=getattr(config, 'visualize', False),
                    only_vis_avg=getattr(config, 'only_vis_avg', False)
                )
                
                avg_reward = inference_reward(
                    model=grm_model,
                    cam_high_frames=high_frame_list,
                    cam_left_frames=left_frame_list,
                    cam_right_frames=right_frame_list,
                    run_roots=rew_cfg.run_roots,
                    cam_dirs_map=rew_cfg.cam_dirs_map,
                    ref_end_map=rew_cfg.ref_end_map,
                    task=rew_cfg.task_instruction,
                    frame_interval=rew_cfg.frame_interval,
                    batch_size=rew_cfg.batch_size,
                    visualize=rew_cfg.visualize,
                    only_vis_avg=rew_cfg.only_vis_avg,
                    gamma=rew_cfg.gamma,
                    completion_reward=completion_rewards
                )
                
                # 更新轨迹中的奖励
                if len(avg_reward) != len(trajectory):
                    print(f"⚠ Reward length mismatch: {len(avg_reward)} vs {len(trajectory)}")
                    avg_reward = np.interp(np.arange(len(trajectory)), 
                                           np.linspace(0, len(trajectory)-1, len(avg_reward)), 
                                           avg_reward)
                
                for i, transition in enumerate(trajectory):
                    transition["rewards"] = avg_reward[i]
                
                if info.get("succeed", False):
                    # 添加 MC returns 和 embeddings
                    trajectory = add_mc_returns_to_trajectory(
                        trajectory, config.discount, FLAGS.reward_scale, FLAGS.reward_bias, 
                        config.reward_neg, is_sparse_reward=False
                    )
                    trajectory = add_embeddings_to_trajectory(trajectory, octo_model, tasks=tasks)
                    trajectory = add_next_embeddings_to_trajectory(trajectory)
                    
                    for transition in trajectory:
                        transitions.append(deepcopy(transition))
                    success_count += 1
                    pbar.update(1)
                    print(f"\n✓ 成功演示 {success_count}/{success_needed}")
                
                trajectory = []
                completion_rewards = []
                returns = 0
                obs, info = env.reset()
                time.sleep(2.0)
                
    except KeyboardInterrupt:
        print("\n\n用户中断，正在保存已收集的数据...")
    
    # 保存数据
    if not os.path.exists("./demo_data"):
        os.makedirs("./demo_data")
    
    if len(transitions) > 0:
        uuid = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"./demo_data/{FLAGS.exp_name}_{success_count}_demos_dopamine_{uuid}.pkl"
        with open(file_name, "wb") as f:
            pkl.dump(transitions, f)
            print(f"✓ 保存 {success_count} 个演示到 {file_name}")


if __name__ == "__main__":
    app.run(main)
