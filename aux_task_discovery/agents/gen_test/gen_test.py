import copy
from typing import Union

import numpy as np
import torch
from torch import optim

from aux_task_discovery.utils import random_argmax
import aux_task_discovery.utils.pytorch_utils as ptu
from aux_task_discovery.agents.dqn import DQNAgent
from aux_task_discovery.models import MasterUserNetwork
from aux_task_discovery.agents.gen_test.generators import get_generator
from aux_task_discovery.agents.gen_test.testers import get_tester


class GenTestAgent(DQNAgent):
    '''
    DQN agent with mutliple output heads for action values corresponding to each of 
    a given number of auxillery tasks. Auxillery tasks are generated by the given
    generator class and their utility is evaluated by the given tester class.
    Algorithm specified in Algorithm 1 @ https://arxiv.org/abs/2210.14361
    '''
    def __init__(
        self,
        input_shape: tuple,
        n_actions: int,
        generator: str,
        tester: str,
        n_aux_tasks = 5,
        age_threshold = 0,
        replace_cycle = 500,
        replace_ratio = 0.2,
        tester_tau = 0.05,
        seed = 42,
        learning_rate = 0.01, 
        adam_beta_1 = 0.9,
        adam_beta_2 = 0.999,
        epsilon = 0.1,
        epsilon_final = 0.1,
        anneal_epsilon = False,
        n_anneal = 10000,
        gamma = 1.0,
        hidden_size = 500,
        activation = 'tanh',
        buffer_size = 500,
        batch_size = 16,
        update_freq = 1,
        target_update_freq=100,
        learning_start = 500,
    ):
        self.n_aux_tasks = n_aux_tasks
        self.age_threshold = age_threshold
        self.replace_cycle = replace_cycle
        self.n_replace = round(n_aux_tasks * replace_ratio)
        self.task_ages = np.zeros(n_aux_tasks)
        self.task_utils = np.zeros(n_aux_tasks)
        super().__init__(
            input_shape=input_shape,
            n_actions=n_actions,
            seed=seed,
            learning_rate=learning_rate,
            adam_beta_1=adam_beta_1,
            adam_beta_2=adam_beta_2,
            epsilon=epsilon,
            epsilon_final=epsilon_final,
            anneal_epsilon=anneal_epsilon,
            n_anneal=n_anneal,
            gamma=gamma,
            n_hidden=1,
            hidden_size=hidden_size,
            activation=activation,
            buffer_size=buffer_size,
            batch_size=batch_size,
            update_freq=update_freq,
            target_update_freq=target_update_freq,
            learning_start=learning_start,
            )
        self.generator = get_generator(generator)(
            input_shape=input_shape,
            seed=seed,
            )
        self.tester = get_tester(tester)(
            model=self.model,
            tau=tester_tau,
            seed=seed
            )
        self.tasks = np.array(self.generator.generate_tasks(self.n_aux_tasks))

    def _setup_model(self):
        self.model = MasterUserNetwork(
                input_shape=self.input_shape,
                n_actions=self.n_actions,
                n_aux_tasks=self.n_aux_tasks,
                hidden_size=self.hidden_size,
                activation=self.activation,
                )

    @torch.no_grad()
    def get_action(self, obs: np.ndarray):
        '''
        Selects e-greedy action using q-values for main task from model
        '''
        if self.rand_gen.rand() < self.epsilon:
            return self.rand_gen.randint(self.n_actions)
        obs = ptu.from_numpy(obs).unsqueeze(0)
        q_vals = ptu.to_numpy(self.model(obs))['main'][0]
        act = random_argmax(q_vals)
        return act
    
    def _step(
        self, 
        obs: np.ndarray, 
        act: int,
        rew: Union[float, int], 
        next_obs: np.ndarray, 
        terminated: bool,
        truncated: bool
    ) -> dict:
        '''
        Adds a single transition to the replay buffer and updates model weights 
        and algorithm parameters as nessesary. Returns dict for logging.
        '''
        # DQN Update
        log_info = super()._step(
                    obs=obs,
                    act=act, 
                    rew=rew, 
                    next_obs=next_obs, 
                    terminated=terminated, 
                    truncated=truncated
                    )

        # Compute task ages and utils
        self.task_ages += 1
        if self.step_idx >= self.learning_start:
            # Compute task utils
            self.task_utils = self.tester.eval_tasks(batch=self.replay_buffer.last_batch, observation=obs)
        task_info = {}
        for i in range(self.n_aux_tasks):
            task_info[f'aux_{i}_age'] = self.task_ages[i]
            task_info[f'aux_{i}_util'] = self.task_utils[i]
        log_info.update(task_info)

        # Gen Test Update
        if self.step_idx >= self.learning_start and self.step_idx % self.replace_cycle == 0:
            self._update_tasks()
            self._update_target_network()
        
        return log_info
    
    def _update_tasks(self):
        '''
        Identify tasks with age > self.age_threshold and replace n_aux_tasks * replace_ratio of them
        with new tasks generated by self.generator. Reset input and output weights of features 
        induced by the replaced tasks and set ages of new tasks to 0.
        '''
        if self.n_replace == 0:
            return
        # Get utils for tasks with with age > self.age_threshold
        utils = self.task_utils[self.task_ages>self.age_threshold]
        if len(utils) > 0:
            # Get idxs in origional task list for tasks in the above slice
            idxs = np.arange(self.n_aux_tasks)[self.task_ages>self.age_threshold]
            # Get idxs of (n_aux_tasks * replace_ratio) tasks with lowest util
            curr_tasks = list(zip(utils, idxs))
            curr_tasks.sort(key=lambda x : x[0])
            idxs = [x[1] for x in curr_tasks[:self.n_replace]]
            # Replace with new tasks
            new_tasks = self.generator.generate_tasks(len(idxs))
            self.tasks[idxs] = new_tasks
            # Reset input and output weights and of features induced by the replaced tasks
            self.model.reset_task_params(idxs)
            # Reset tester as needed for replaced tasks. If tester uses a trace, 
            # will reset traces for features induced by replaced tasks
            self.tester.reset_tasks(idxs)
            # Set ages to 0 for new tasks
            self.task_ages[idxs] = 0

    def _get_losses(self, batch: dict):
        '''
        Computes squarred TD error for each transition in batch for each output head.
        '''
        obs = batch["observations"]
        act = batch["actions"]
        rew = batch["rewards"]
        next_obs = batch["next_observations"]
        terminated = batch["terminateds"]
        truncated = batch["truncateds"]

        # Dict with action-value estimates for obs for each output head from model
        preds_all = self.model(ptu.from_numpy(obs))
        # Dict with action-value estimates for next_obs for each output head from target model
        next_q_all = self.target_model(ptu.from_numpy(next_obs))

        # Get loss for main task
        losses = {}
        preds = preds_all['main'][torch.arange(obs.shape[0]), ptu.from_numpy(act)]
        next_qs = next_q_all['main'].max(dim=-1)[0].detach()
        next_qs[terminated & ~truncated] = 0
        targets = ptu.from_numpy(rew) + (self.gamma * next_qs)
        main_loss = ((targets - preds) ** 2)
        losses['main'] = main_loss
        
        # Get loss for aux tasks
        for i, task in enumerate(self.tasks):
            preds = preds_all[i][torch.arange(obs.shape[0]), ptu.from_numpy(act)]
            next_qs = next_q_all[i].max(dim=-1)[0].detach()
            gammas = ptu.from_numpy(task.gamma(next_obs))
            cumulants = ptu.from_numpy(task.cumulant(next_obs))
            targets =  cumulants + (gammas * next_qs)
            aux_loss = ((targets - preds) ** 2)
            losses[i] = aux_loss
        
        return losses

    def train(self):
        '''
        Computes mean squared TD error on batch from replay buffer for each output head. 
        Sums loss for each output head to compute total loss and updates model weights. 
        Returns a dict containing loss metrics.
        '''
        self.model.train()
        batch = self.replay_buffer.sample(batch_size=self.batch_size)
        losses = self._get_losses(batch)
        loss_info = {}
        loss = losses['main'].mean()
        loss_info['main_loss'] = loss.item()
        for i in range(self.n_aux_tasks):
            aux_loss = losses[i].mean()
            loss += aux_loss
            loss_info[f'aux_{i}_loss'] = aux_loss.item()
        loss_info['total_loss'] = loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss_info