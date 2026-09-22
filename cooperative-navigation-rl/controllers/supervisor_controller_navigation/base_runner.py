#负责通用初始化和统一接口

import os
import numpy as np
# import wandb
import torch
from tensorboardX import SummaryWriter
import pandas as pd

from AC.model import Model
from supervisor_controller_navigation import Epuck2Supervisor


class BaseRunner(object):
    """Base class for training recurrent policies."""

    def __init__(self, config):
        """
        Base class for training recurrent policies.
        :param config: (dict) Config dictionary containing parameters for training.
        """
        self.args = config["args"]#把所有的配置参数都放进来
        self.device = config["device"]
        self.algorithm_name = self.args.algorithm_name #sac
        self.env_name = self.args.env_name #navigation导航
        self.max_episodes = self.args.max_episodes #8000
        self.buffer_length = self.args.buffer_length #800 单个 episode 的最大长度
        self.reward_normalize = self.args.reward_normalize #是否对奖励进行归一化
        self.batch_size = self.args.batch_size #128
        self.use_eval = self.args.use_eval #是否在训练过程中进行性能评估
        self.gamma = self.args.gamma #0.99 折扣因子



        self.total_train_steps = 0
        self.total_env_steps = 0
        self.num_agents = config["num_agents"]
        self.num_envs = self.args.n_rollout_threads #1个环境
        self.actions = None
        self.agent_ids = [i for i in range(self.num_agents)]
        self.train_count, self.eval_count = 0, 0 #记录已执行的训练和评估回合数

        self.env: Epuck2Supervisor = config["env"] #环境实例

        # dir
        self.model_dir = self.args.model_dir
        self.run_dir = config["run_dir"]
        self.log_dir = str(self.run_dir / 'logs')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.tb_writer = SummaryWriter(self.log_dir)

        #为每个智能体（worker）创建单独的 TensorBoard 记录器
        self.tb_train_writers = []
        for i in range(self.num_agents):
            tb_train_dir = os.path.join(self.log_dir, f"worker_{i}")
            if not os.path.exists(tb_train_dir):
                os.makedirs(tb_train_dir)
            self.tb_train_writers.append(SummaryWriter(tb_train_dir))

        #保存轨迹图
        self.image_dir = str(self.run_dir / 'images')
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

        #保存模型和评估结果
        self.save_dir = str(self.run_dir / 'models')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        self.eval_dir = str(self.run_dir / 'eval')
        if not os.path.exists(self.eval_dir):
            os.makedirs(self.eval_dir)



        self.model = Model(#创建一个模型的实例
            self.env.num_observations, self.env.num_states, self.env.num_actions,
            self.device, self.args, self.args.reward_normalize, continuous=True,
            manifold_projector=getattr(self.env, "manifold_projector", None),
            world_size=getattr(self.env, "world_size", None),
            max_wheel_speed=getattr(self.env, "max_speed", None),
        )
        if self.model_dir is not None:
            load_model = torch.load(self.model_dir, map_location=self.device)
            if isinstance(load_model, dict):
                load_result = self.model.load_state_dict(load_model, strict=False)
                if load_result.missing_keys or load_result.unexpected_keys:
                    print(
                        "loaded checkpoint with unmatched keys: "
                        f"missing={load_result.missing_keys}, "
                        f"unexpected={load_result.unexpected_keys}"
                    )
            else:
                load_result = self.model.load_state_dict(
                    load_model.state_dict(), strict=False
                )
                if load_result.missing_keys or load_result.unexpected_keys:
                    print(
                        "loaded checkpoint with unmatched keys: "
                        f"missing={load_result.missing_keys}, "
                        f"unexpected={load_result.unexpected_keys}"
                    )
            print("load model successfully")
        self.optimizer = None


    def run(self):
        """Collect a training episode and perform appropriate training, saving, logging, and evaluation steps."""
        # collect data
        raise NotImplementedError

    def log(self):
        """Log relevent training and rollout colleciton information.."""
        raise NotImplementedError

    def log_clear(self):
        """Clear logging variables so they do not contain stale information."""
        raise NotImplementedError

    def log_env(self, env_info, total_env_steps, suffix=None):
        """
        Log information related to the environment.
        :param env_info: (dict) contains logging information related to the environment.
        :param suffix: (str) optional string to add to end of keys in env_info when logging. 
        """
        raise NotImplementedError
        
    def collect_rollout(self):#采样一条或多条轨迹并存入经验缓冲区
        """Collect a rollout and store it in the buffer."""
        raise NotImplementedError
