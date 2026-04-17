#!/usr/bin/env python3
"""
Pyro5 client wrapper for remote environment.
Use this on the A800 machine to access the environment on the local machine.
"""

import Pyro5.api
import numpy as np
from gymnasium import spaces
from gymnasium import Env
from typing import Optional, Tuple, Dict, Any


class Pyro5EnvWrapper(Env):
    """
    Wrapper to make a Pyro5 remote environment look like a local gym environment.
    This allows the actor on A800 to call env.step() and env.reset() remotely.
    """
    
    def __init__(self, uri: str):
        """
        Args:
            uri: Pyro5 URI of the remote environment (e.g., "PYRO:remote_env@192.168.1.100:9090")
        """
        self.uri = uri
        self.remote_env = Pyro5.api.Proxy(uri)
        
        # Get action and observation spaces from remote
        action_space_dict = self.remote_env.get_action_space()
        self.action_space = self._dict_to_space(action_space_dict)
        
        obs_space_dict = self.remote_env.get_observation_space()
        self.observation_space = self._dict_to_space(obs_space_dict)
        
        print(f"Connected to remote environment at {uri}")
        print(f"Remote action space: {self.action_space}")
        print(f"Remote observation space: {self.observation_space}")
    
    def _dict_to_space(self, space_dict: Dict) -> spaces.Space:
        """Convert serialized space dict back to gymnasium Space object"""
        if space_dict["type"] == "Box":
            return spaces.Box(
                low=space_dict["low"],
                high=space_dict["high"],
                shape=space_dict["shape"],
                dtype=np.dtype(space_dict["dtype"])
            )
        elif space_dict["type"] == "Dict":
            spaces_dict = {}
            for key, space_dict_val in space_dict["spaces"].items():
                spaces_dict[key] = self._dict_to_space(space_dict_val)
            return spaces.Dict(spaces_dict)
        else:
            raise NotImplementedError(f"Space type {space_dict['type']} not supported")
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset the remote environment"""
        obs, info = self.remote_env.reset(**kwargs)
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action in remote environment"""
        obs, reward, done, truncated, info = self.remote_env.step(action)
        return obs, reward, done, truncated, info
    
    def close(self):
        """Clean up Pyro5 connection"""
        self.remote_env._pyroRelease()


def create_pyro_env(local_machine_ip: str, port: int = 9090) -> Pyro5EnvWrapper:
    """
    Convenience function to create a Pyro5 environment wrapper.
    
    Args:
        local_machine_ip: IP address of the local machine running env_server.py
        port: Port number (default: 9090)
    
    Returns:
        Pyro5EnvWrapper instance
    """
    uri = f"PYRO:remote_env@{local_machine_ip}:{port}"
    return Pyro5EnvWrapper(uri)


if __name__ == "__main__":
    # Test the wrapper
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Pyro5 Environment Wrapper")
    parser.add_argument("--ip", type=str, required=True, help="IP of local machine")
    parser.add_argument("--port", type=int, default=9090, help="Port of Pyro5 server")
    args = parser.parse_args()
    
    print(f"Connecting to remote environment at {args.ip}:{args.port}...")
    env = create_pyro_env(args.ip, args.port)
    
    print("\nTesting reset...")
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape if hasattr(obs, 'shape') else {k: v.shape for k, v in obs.items()}}")
    
    print("\nTesting step with random action...")
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print(f"Reward: {reward}, Done: {done}")
    
    print("\nTest successful!")
    env.close()
