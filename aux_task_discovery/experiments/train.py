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

    # Set experiment seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    # Set pytorch device
    ptu.init_gpu(use_gpu=config.use_gpu, gpu_id=config.gpu_id)

    # Make env
    env = gym.make(config.env, seed=config.seed)
    assert isinstance(env.action_space, gym.spaces.Discrete), 'DQN requires discrete action space'
    
    agent_args = config.agent_args.copy()
    # If normalizing input, get low and high bounds for observation space
    if config.normalize_input:
        assert isinstance(env.observation_space, gym.spaces.Box), 'Normalizing input currently only supports Box observation space'
        agent_args['obs_bound'] = (env.observation_space.low[0], env.observation_space.high[0])
    
    # Make agent
    agent = get_agent(config.agent)(
        env=env,
        seed=config.seed,
        **agent_args
        )
    
    # Track model parameters and gradients
    wandb.watch(agent.model, log='all')

    # Train loop
    obs, _ = env.reset()
    episode_idx = 0
    episode_reward = 0
    episode_len = 0
    step_idx = 0
    while step_idx < config.max_steps and episode_idx < config.max_episodes:
        # Start episode timer
        if episode_len == 0:
            episode_start_time = time.time()
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
        step_idx += 1
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
            episode_idx += 1
            # Print episode metrics
            if episode_idx <= 10 or episode_idx % 10 == 0:
                print(f"Step: {step_idx}, Episode: {episode_idx}, Episode Reward: {episode_reward}, Episode Length: {episode_len}, Episode Runtime: {time.time() - episode_start_time}")
            # Reset episode metrics
            episode_len = 0
            episode_reward = 0
        
        # Log step and episode metrics
        wandb.log(step_log)
    
    wandb.finish()






