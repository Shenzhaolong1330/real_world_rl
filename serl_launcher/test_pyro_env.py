#!/usr/bin/env python3
"""
Test script to verify Pyro5 environment wrapper works correctly.
Run this on the A800 machine after starting env_server.py on the local machine.
"""

import numpy as np
import argparse
from serl_launcher.utils.pyro_env_wrapper import create_pyro_env


def test_pyro_env(ip: str, port: int):
    """Test Pyro5 environment connection and basic operations"""
    
    print(f"\n{'='*60}")
    print(f"Testing Pyro5 Environment Connection")
    print(f"{'='*60}\n")
    
    # Create Pyro5 environment
    print(f"Connecting to remote environment at {ip}:{port}...")
    env = create_pyro_env(ip, port)
    
    # Test 1: Check action space
    print(f"\n[Test 1] Action Space:")
    print(f"  Shape: {env.action_space.shape}")
    print(f"  Low: {env.action_space.low}")
    print(f"  High: {env.action_space.high}")
    
    # Test 2: Check observation space
    print(f"\n[Test 2] Observation Space:")
    if hasattr(env.observation_space, 'spaces'):
        for key, space in env.observation_space.spaces.items():
            print(f"  {key}: shape={space.shape}, dtype={space.dtype}")
    else:
        print(f"  Shape: {env.observation_space.shape}")
    
    # Test 3: Reset environment
    print(f"\n[Test 3] Reset Environment:")
    obs, info = env.reset()
    print(f"  Observation type: {type(obs)}")
    if isinstance(obs, dict):
        print(f"  Observation keys: {obs.keys()}")
        for key, val in obs.items():
            if hasattr(val, 'shape'):
                print(f"    {key}: shape={val.shape}, dtype={val.dtype}")
    else:
        print(f"  Observation shape: {obs.shape}")
    print(f"  Info keys: {info.keys()}")
    
    # Test 4: Random action
    print(f"\n[Test 4] Random Action:")
    action = env.action_space.sample()
    print(f"  Action shape: {action.shape}")
    print(f"  Action: {action}")
    
    # Test 5: Step environment
    print(f"\n[Test 5] Step Environment:")
    obs, reward, done, truncated, info = env.step(action)
    print(f"  Reward: {reward}")
    print(f"  Done: {done}")
    print(f"  Truncated: {truncated}")
    print(f"  Info keys: {info.keys()}")
    if 'succeed' in info:
        print(f"  Succeed: {info['succeed']}")
    
    # Test 6: Multiple steps
    print(f"\n[Test 6] Multiple Steps:")
    obs, _ = env.reset()
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"  Step {i+1}: reward={reward:.3f}, done={done}")
        if done:
            print(f"  Episode finished at step {i+1}")
            break
    
    # Test 7: Check info dict serialization
    print(f"\n[Test 7] Info Dict Serialization:")
    obs, _ = env.reset()
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        if done:
            print(f"  Episode done!")
            print(f"  Info keys: {list(info.keys())}")
            for key, val in info.items():
                print(f"    {key}: {type(val)}")
                if isinstance(val, dict):
                    print(f"      Sub-keys: {list(val.keys())}")
            break
    
    print(f"\n{'='*60}")
    print(f"All tests passed!")
    print(f"{'='*60}\n")
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Pyro5 Environment Wrapper")
    parser.add_argument("--ip", type=str, required=True, help="IP of local machine")
    parser.add_argument("--port", type=int, default=9090, help="Port of Pyro5 server")
    args = parser.parse_args()
    
    test_pyro_env(args.ip, args.port)
