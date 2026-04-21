#!/usr/bin/env python3
"""
Pyro5 server for remote environment access.
Run this on the local machine with Franka robot and RealSense cameras.
"""

import Pyro5.api
import Pyro5.errors
import numpy as np
from gymnasium import spaces
import argparse
import sys
import os
import base64
import importlib

# Register numpy array serializer for Pyro5
from Pyro5.serializers import SerpentSerializer

def numpy_to_dict(arr):
    """Convert numpy array to dict for serialization"""
    return {
        '__class__': 'numpy.ndarray',
        'data': arr.tobytes(),
        'dtype': str(arr.dtype),
        'shape': arr.shape
    }

def dict_to_numpy(classname, data):
    """Convert dict back to numpy array"""
    # Handle serpent's bytes serialization
    if isinstance(data['data'], str):
        # Serpent encodes bytes as base64 string
        byte_data = base64.b64decode(data['data'])
    elif isinstance(data['data'], dict) and 'data' in data['data']:
        # Nested serialization
        byte_data = base64.b64decode(data['data']['data'])
    elif isinstance(data['data'], (bytes, bytearray)):
        byte_data = data['data']
    else:
        # Try to convert list to bytes
        byte_data = bytes(data['data']) if isinstance(data['data'], list) else data['data']
    
    return np.frombuffer(byte_data, dtype=data['dtype']).reshape(data['shape'])

# Register the custom serializer
SerpentSerializer.register_class_to_dict(np.ndarray, numpy_to_dict)
SerpentSerializer.register_dict_to_class('numpy.ndarray', dict_to_numpy)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_get_environment(task_name: str):
    module_name = f"examples.experiments.{task_name}.config"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            f"Failed to import task config module '{module_name}'. "
            "Please check --task and ensure the task folder exists under examples/experiments/."
        ) from error

    if not hasattr(module, "get_environment"):
        raise AttributeError(
            f"Module '{module_name}' does not define 'get_environment'."
        )
    return module.get_environment


@Pyro5.api.expose
class RemoteEnv:
    """Wrapper to expose gym environment via Pyro5"""
    
    def __init__(self, env):
        self.env = env
        
    def reset(self, **kwargs):
        """Reset environment and return initial observation"""
        obs, info = self.env.reset(**kwargs)
        # Convert to serializable format (numpy arrays are already serializable by Pyro5)
        return obs, info
    
    def step(self, action):
        """Execute action and return (obs, reward, done, truncated, info)"""
        # Deserialize action if needed (Pyro5 handles numpy arrays automatically)
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Remove non-serializable items from info dict
        if "left" in info:
            info.pop("left")
        if "right" in info:
            info.pop("right")
            
        return obs, reward, done, truncated, info
    
    def get_action_space_sample(self):
        """Sample random action from action space"""
        return self.env.action_space.sample()
    
    def get_action_space(self):
        """Return action space definition"""
        # Convert action space to dict for serialization
        if isinstance(self.env.action_space, spaces.Box):
            return {
                "type": "Box",
                "low": self.env.action_space.low,
                "high": self.env.action_space.high,
                "shape": self.env.action_space.shape,
                "dtype": str(self.env.action_space.dtype)
            }
        else:
            raise NotImplementedError(f"Action space type {type(self.env.action_space)} not supported")
    
    def get_observation_space(self):
        """Return observation space definition"""
        # Convert observation space to dict for serialization
        if isinstance(self.env.observation_space, spaces.Dict):
            obs_space_dict = {}
            for key, space in self.env.observation_space.spaces.items():
                if isinstance(space, spaces.Box):
                    obs_space_dict[key] = {
                        "type": "Box",
                        "low": space.low,
                        "high": space.high,
                        "shape": space.shape,
                        "dtype": str(space.dtype)
                    }
            return {"type": "Dict", "spaces": obs_space_dict}
        else:
            raise NotImplementedError(f"Observation space type {type(self.env.observation_space)} not supported")


def main():
    parser = argparse.ArgumentParser(description="Pyro5 Environment Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=9090, help="Port to listen on")
    parser.add_argument("--task", type=str, default="task2_insert_vial", 
                        help="Task config to use (e.g., task2_insert_vial)")
    args = parser.parse_args()
    
    print(f"Creating environment for task: {args.task}")
    get_environment = load_get_environment(args.task)
    
    # Create the real environment (not fake_env)
    env = get_environment(
        fake_env=False,  # Real environment with hardware
        save_video=False,
        classifier=True,
        stack_obs_num=2
    )
    
    print(f"Environment created successfully")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Wrap environment for Pyro5
    remote_env = RemoteEnv(env)
    
    # Start Pyro5 daemon
    daemon = Pyro5.api.Daemon(host=args.host, port=args.port)
    
    # Register the remote environment
    uri = daemon.register(remote_env, "remote_env")
    
    print(f"\n{'='*60}")
    print(f"Pyro5 Environment Server Started")
    print(f"{'='*60}")
    print(f"URI: {uri}")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"\nOn A800 machine, use this URI to connect:")
    print(f"  PYRO:remote_env@<LOCAL_MACHINE_IP>:{args.port}")
    print(f"{'='*60}\n")
    
    # Request loop
    daemon.requestLoop()


if __name__ == "__main__":
    main()
