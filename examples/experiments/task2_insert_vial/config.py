import os
import jax
import numpy as np
import jax.numpy as jnp

try:
    from franka_env.envs.wrappers import (
        Quat2EulerWrapper,
        SpacemouseIntervention,
        MultiCameraBinaryRewardClassifierWrapper,
    )
    from franka_env.envs.relative_env import RelativeFrame
    from franka_env.envs.franka_env import DefaultEnvConfig
except ModuleNotFoundError:
    from serl_robot_infra.franka_env.envs.wrappers import (
        Quat2EulerWrapper,
        SpacemouseIntervention,
        MultiCameraBinaryRewardClassifierWrapper,
    )
    from serl_robot_infra.franka_env.envs.relative_env import RelativeFrame
    from serl_robot_infra.franka_env.envs.franka_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from ..config import DefaultTrainingConfig
from .wrapper import InsertVialEnv, GripperPenaltyWrapper


class EnvConfig(DefaultEnvConfig):
    # For remote deployment, change to robot machine IP
    # Example: SERVER_URL = "http://192.168.1.100:5000/"
    SERVER_URL: str = "http://127.0.0.2:5000/"  # ⚠️ Change this for remote deployment
    REALSENSE_CAMERAS = {
        "wrist_1": {
            "serial_number": "344322074412",
            "dim": (1280, 720),
            "exposure": 10500,
        },
        "side_policy_256": {
            "serial_number": "401622073044",
            "dim": (1280, 720),
            "exposure": 13000,
        },
        "side_classifier": {
            "serial_number": "401622073044", 
            "dim": (1280, 720),
            "exposure": 13000,
        },
        "demo": {
            "serial_number": "401622073044", 
            "dim": (1280, 720),
            "exposure": 13000,
        },
    }
    IMAGE_CROP = {"wrist_1": lambda img: img,
                  "side_policy_256": lambda img: img[400:680, 400:1000],
                  "side_classifier": lambda img: img[400:680, 400:1000],
                  "demo": lambda img: img[0:720, 0:1280]}

    TARGET_POSE = np.array([0.400, 0.138, 0.241, np.pi, 0, np.pi/2])
    RESET_POSE = np.array([0.400, 0.138, 0.341, np.pi, 0, np.pi/2])
    ACTION_SCALE = np.array([0.05, 0.2, 1])
    RANDOM_RESET = True
    DISPLAY_IMAGE = True
    RANDOM_XY_RANGE = 0.05
    RANDOM_RZ_RANGE = 0.05
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.1, 0.1, 0.1, 0.01, 0.01, 0.3])
    ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([0.1, 0.1, 0.05, 0.01, 0.01, 0.3])
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.008,
        "translational_clip_y": 0.005, 
        "translational_clip_z": 0.005,
        "translational_clip_neg_x": 0.008,
        "translational_clip_neg_y": 0.005,
        "translational_clip_neg_z": 0.005, 
        "rotational_clip_x": 0.02,
        "rotational_clip_y": 0.02,
        "rotational_clip_z": 0.02,
        "rotational_clip_neg_x": 0.02,
        "rotational_clip_neg_y": 0.02,
        "rotational_clip_neg_z": 0.02, 
        "rotational_Ki": 0,
    }  # for normal operation other than reset procedure
    PRECISION_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0.0,
        "translational_clip_x": 0.01,
        "translational_clip_y": 0.01,
        "translational_clip_z": 0.01,
        "translational_clip_neg_x": 0.01,
        "translational_clip_neg_y": 0.01,
        "translational_clip_neg_z": 0.01,
        "rotational_clip_x": 0.03,
        "rotational_clip_y": 0.03,
        "rotational_clip_z": 0.03,
        "rotational_clip_neg_x": 0.03,
        "rotational_clip_neg_y": 0.03,
        "rotational_clip_neg_z": 0.03,
        "rotational_Ki": 0.0,
    }  # only for reset procedure
    MAX_EPISODE_LENGTH = 100
 

class TrainConfig(DefaultTrainingConfig):
    image_keys = ["side_policy_256", "wrist_1"]
    classifier_keys = ["side_classifier"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    checkpoint_period = 2000
    cta_ratio = 2
    random_steps = 0
    discount = 0.98
    buffer_period = 1000
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"
    reward_neg = -0.05
    task_desc = "Insert the vial into the rack"
    octo_path = "/home/szl/real_world_rl/octo-small"
    
    # 稠密奖励配置
    use_dense_reward = True  # 是否使用稠密奖励
    success_reward = 10.0    # 成功奖励值
    success_threshold = 0.9  # 分类器阈值
    
    # GRM 稠密奖励配置
    use_grm_reward = False  # 是否使用 GRM 稠密奖励（优先级高于分类器）
    weight_path = "/home/szl/real_world_rl/weights/Robo-Dopamine-GRM-2.0-8B-Preview"  # GRM 模型路径
    frame_interval = 4  # GRM 推理的帧间隔
    batch_dopamine = 45  # GRM 批处理大小
    visualize = False  # 是否可视化 GRM 奖励
    only_vis_avg = False  # 是否只可视化平均奖励

    def get_environment(self, fake_env=False, save_video=False, classifier=False, stack_obs_num=1):
        env = InsertVialEnv(fake_env=fake_env, save_video=save_video, config=EnvConfig())
        if not fake_env:
            env = SpacemouseIntervention(env)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=stack_obs_num, act_exec_horizon=None)
        
        # 添加稠密奖励分类器
        if classifier and self.use_dense_reward:
            classifier_func = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                logit = classifier_func(obs)[0]
                if sigmoid(logit) > self.success_threshold:
                    return self.success_reward
                else:
                    return self.reward_neg

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        
        env = GripperPenaltyWrapper(env, penalty=-0.2)
        return env


def get_environment(fake_env=False, save_video=False, classifier=False, stack_obs_num=1):
    return TrainConfig().get_environment(
        fake_env=fake_env,
        save_video=save_video,
        classifier=classifier,
        stack_obs_num=stack_obs_num,
    )
