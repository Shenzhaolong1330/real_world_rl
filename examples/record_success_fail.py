"""
成功/失败状态数据收集脚本

功能说明：
- 收集用于训练奖励分类器的数据
- 通过键盘（双击'.'键）标记成功状态
- 同时保存成功和失败的转移数据

输出：
  - classifier_data/<exp_name>_<num>_success_images_<timestamp>.pkl: 成功状态
  - classifier_data/<exp_name>_failure_images_<timestamp>.pkl: 失败状态

使用方式：
  python record_success_fail.py --exp_name=<实验名称> --successes_needed=200
  
操作说明：
  - 遥操作机器人执行任务
  - 当达到成功状态时，快速双击'.'键标记
  - 收集足够多的成功样本后自动保存
"""
import copy
import os
from tqdm import tqdm
import numpy as np
import pickle as pkl
import datetime
from absl import app, flags

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "task2_insert_vial", "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 200, "Number of successful transitions to collect.")
flags.DEFINE_boolean("use_pyro_env", False, "Use Pyro5 remote environment")
flags.DEFINE_string("pyro_env_ip", None, "IP address of the Pyro5 environment server")
flags.DEFINE_integer("pyro_env_port", 9090, "Port of the Pyro5 environment server")


def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    
    # Create environment based on mode
    if FLAGS.use_pyro_env:
        from serl_launcher.utils.pyro_env_wrapper import create_pyro_env
        print(f"Connecting to Pyro5 remote environment at {FLAGS.pyro_env_ip}:{FLAGS.pyro_env_port}")
        env = create_pyro_env(FLAGS.pyro_env_ip, FLAGS.pyro_env_port)
    else:
        env = config.get_environment(fake_env=False, save_video=False, classifier=False, stack_obs_num=2)
    
    # Wrap with CompletionRewardClassifierWrapper for keyboard input
    from franka_env.envs.wrappers import CompletionRewardClassifierWrapper
    env = CompletionRewardClassifierWrapper(env)

    obs, _ = env.reset()
    successes = []
    failures = []
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    
    print("\n" + "="*60)
    print("操作说明:")
    print("  - 使用 spacemouse 遥操作机器人执行任务")
    print("  - 当达到成功状态时，快速双击 '.' 键标记成功")
    print("  - 按 Ctrl+C 可提前退出")
    print("="*60 + "\n")
    
    try:
        while len(successes) < success_needed:
            actions = np.zeros(env.action_space.sample().shape) 
            next_obs, rew, done, truncated, info = env.step(actions)
            if "intervene_action" in info:
                actions = info["intervene_action"]

            transition = copy.deepcopy(
                dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=rew,
                    masks=1.0 - done,
                    dones=done,
                )
            )
            obs = next_obs
            
            # Check if marked as success (via keyboard double-press '.')
            if info.get('succeed', False) or rew > 0.5:
                successes.append(transition)
                pbar.update(1)
                print(f"\n✓ 成功样本 {len(successes)}/{success_needed}")
            else:
                failures.append(transition)

            if done or truncated:
                obs, _ = env.reset()
                
    except KeyboardInterrupt:
        print("\n\n用户中断，正在保存已收集的数据...")
    
    # Save data
    if not os.path.exists("./classifier_data"):
        os.makedirs("./classifier_data")
    
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    if len(successes) > 0:
        file_name = f"./classifier_data/{FLAGS.exp_name}_{len(successes)}_success_images_{uuid}.pkl"
        with open(file_name, "wb") as f:
            pkl.dump(successes, f)
            print(f"✓ 保存 {len(successes)} 个成功样本到 {file_name}")

    if len(failures) > 0:
        file_name = f"./classifier_data/{FLAGS.exp_name}_failure_images_{uuid}.pkl"
        with open(file_name, "wb") as f:
            pkl.dump(failures, f)
            print(f"✓ 保存 {len(failures)} 个失败样本到 {file_name}")
        
if __name__ == "__main__":
    app.run(main)
