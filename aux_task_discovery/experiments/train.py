import time
import random

import wandb
import tqdm
import numpy as np
import torch
import gymnasium as gym

import aux_task_discovery.utils.pytorch_utils as ptu
from aux_task_discovery.utils.constants import WANDB_PROJECT
from aux_task_discovery.agents import get_agent


def training_loop(args=None):
    # Start W&B run
    wandb.init(project=WANDB_PROJECT, config=args)
    config = wandb.config

    # Set expirement seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    # Set pytorch device
    ptu.init_gpu(use_gpu=config.use_gpu, gpu_id=config.gpu_id)

    # Make env
    env = gym.make(config.env, seed=config.seed)
    assert isinstance(env.action_space, gym.spaces.Discrete), 'DQN requires discrete action space'
    
    # Make agent
    agent = get_agent(config.agent)(
        input_shape=env.observation_space.shape,
        n_actions=env.action_space.n,
        seed=config.seed,
        **config.agent_args
        )

    # Train loop
    obs, _ = env.reset()
    episode_idx = 0
    episode_reward = 0
    episode_len = 0
    for step_idx in tqdm.trange(config.n_steps, dynamic_ncols=True):
        # Get action from agent
        act = agent.get_action(obs)
        # Step env with agent action
        next_obs, rew, terminated, truncated, _ = env.step(act)
        # Step agent with single transition and get log data
        step_log = agent.step(
                        obs=obs,
                        act=act, 
                        rew=rew,
                        next_obs=next_obs,
                        terminated=terminated,
                        truncated=truncated,
                        )
        step_log.update({
            'env_step': step_idx,
            'reward': rew,
        })
        episode_reward += rew
        episode_len += 1
        obs = next_obs

        if terminated:
            # Reset env
            obs, _ = env.reset()
            # Add episode metrics
            step_log.update({
                'episode': episode_idx,
                'episode_reward': episode_reward,
                'episode_len': episode_len,
            })
            episode_len = 0
            episode_reward = 0
            episode_idx += 1
            
        
        # Log metrics
        wandb.log(step_log)






