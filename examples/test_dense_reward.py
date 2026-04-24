#!/usr/bin/env python3
"""
测试稠密奖励流程

测试步骤：
1. 测试环境创建
2. 测试 CompletionRewardClassifierWrapper
3. 测试分类器加载
4. 测试稠密奖励计算
"""
import os
import sys
import jax
import numpy as np
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "task2_insert_vial", "Name of experiment to test.")
flags.DEFINE_boolean("test_wrapper", True, "Test CompletionRewardClassifierWrapper.")
flags.DEFINE_boolean("test_classifier", False, "Test reward classifier loading.")


def test_environment_creation():
    """测试环境创建"""
    print("\n" + "="*60)
    print("测试 1: 环境创建")
    print("="*60)
    
    from experiments.mappings import CONFIG_MAPPING
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    
    # 测试 fake_env 模式
    env = config.get_environment(fake_env=True, save_video=False, classifier=False, stack_obs_num=2)
    print(f"✓ Fake environment created successfully")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    
    # 测试 reset
    obs, info = env.reset()
    print(f"✓ Environment reset successfully")
    print(f"  Observation keys: {obs.keys()}")
    
    # 测试 step
    action = env.action_space.sample()
    next_obs, reward, done, truncated, info = env.step(action)
    print(f"✓ Environment step successfully")
    print(f"  Reward: {reward}, Done: {done}")
    
    env.close()
    return True


def test_completion_wrapper():
    """测试 CompletionRewardClassifierWrapper"""
    print("\n" + "="*60)
    print("测试 2: CompletionRewardClassifierWrapper")
    print("="*60)
    
    try:
        from experiments.mappings import CONFIG_MAPPING
        from franka_env.envs.wrappers import CompletionRewardClassifierWrapper
        
        config = CONFIG_MAPPING[FLAGS.exp_name]()
        env = config.get_environment(fake_env=True, save_video=False, classifier=False, stack_obs_num=2)
        
        # 添加 wrapper
        env = CompletionRewardClassifierWrapper(env)
        print(f"✓ CompletionRewardClassifierWrapper added successfully")
        print(f"  操作说明: 快速双击 '.' 键标记成功")
        
        # 测试 reset
        obs, info = env.reset()
        print(f"✓ Wrapped environment reset successfully")
        print(f"  Info keys: {info.keys()}")
        
        # 测试 step
        action = env.action_space.sample()
        next_obs, reward, done, truncated, info = env.step(action)
        print(f"✓ Wrapped environment step successfully")
        print(f"  Reward: {reward}, Done: {done}, Succeed: {info.get('succeed', False)}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_classifier_loading():
    """测试分类器加载"""
    print("\n" + "="*60)
    print("测试 3: 奖励分类器加载")
    print("="*60)
    
    try:
        from experiments.mappings import CONFIG_MAPPING
        from serl_launcher.networks.reward_classifier import load_classifier_func
        
        config = CONFIG_MAPPING[FLAGS.exp_name]()
        env = config.get_environment(fake_env=True, save_video=False, classifier=False, stack_obs_num=2)
        
        # 检查分类器 checkpoint 是否存在
        ckpt_path = os.path.abspath("classifier_ckpt/")
        if not os.path.exists(ckpt_path):
            print(f"⚠ Classifier checkpoint not found at {ckpt_path}")
            print(f"  请先运行: python train_reward_classifier.py --exp_name={FLAGS.exp_name}")
            return False
        
        # 加载分类器
        classifier = load_classifier_func(
            key=jax.random.PRNGKey(0),
            sample=env.observation_space.sample(),
            image_keys=config.classifier_keys,
            checkpoint_path=ckpt_path,
        )
        print(f"✓ Classifier loaded successfully from {ckpt_path}")
        
        # 测试分类器推理
        obs = env.observation_space.sample()
        logit = classifier(obs)
        sigmoid = lambda x: 1 / (1 + jax.numpy.exp(-x))
        prob = sigmoid(logit[0])
        print(f"✓ Classifier inference successful")
        print(f"  Logit: {logit[0]:.4f}, Probability: {prob:.4f}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dense_reward_env():
    """测试稠密奖励环境"""
    print("\n" + "="*60)
    print("测试 4: 稠密奖励环境")
    print("="*60)
    
    try:
        from experiments.mappings import CONFIG_MAPPING
        
        config = CONFIG_MAPPING[FLAGS.exp_name]()
        
        # 检查是否启用稠密奖励
        if not hasattr(config, 'use_dense_reward') or not config.use_dense_reward:
            print(f"⚠ Dense reward not enabled in config")
            return False
        
        # 创建带分类器的环境
        env = config.get_environment(fake_env=True, save_video=False, classifier=True, stack_obs_num=2)
        print(f"✓ Dense reward environment created successfully")
        
        # 测试 reset
        obs, info = env.reset()
        print(f"✓ Environment reset successfully")
        
        # 测试 step
        action = env.action_space.sample()
        next_obs, reward, done, truncated, info = env.step(action)
        print(f"✓ Environment step successfully")
        print(f"  Reward: {reward:.4f}, Done: {done}, Succeed: {info.get('succeed', False)}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main(_):
    print("\n" + "="*60)
    print("稠密奖励流程测试")
    print("="*60)
    
    results = {}
    
    # 测试 1: 环境创建
    results['environment_creation'] = test_environment_creation()
    
    # 测试 2: CompletionRewardClassifierWrapper
    if FLAGS.test_wrapper:
        results['completion_wrapper'] = test_completion_wrapper()
    
    # 测试 3: 分类器加载
    if FLAGS.test_classifier:
        results['classifier_loading'] = test_classifier_loading()
        
        # 测试 4: 稠密奖励环境
        if results['classifier_loading']:
            results['dense_reward_env'] = test_dense_reward_env()
    
    # 打印结果摘要
    print("\n" + "="*60)
    print("测试结果摘要")
    print("="*60)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:30s}: {status}")
    
    # 计算通过率
    passed = sum(results.values())
    total = len(results)
    print(f"\n通过率: {passed}/{total} ({100*passed/total:.1f}%)")
    
    if passed == total:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print("\n⚠ 部分测试失败，请检查错误信息")
        return 1


if __name__ == "__main__":
    app.run(main)
