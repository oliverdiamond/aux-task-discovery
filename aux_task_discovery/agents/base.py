import numpy as np

class BaseAgent():
    def __init__(self, seed: int) -> None:
        self.rand_gen = np.random.RandomState(seed)
        self.step_idx = 1

    def get_action(self, obs):
        raise NotImplementedError()

    def step(self, 
        obs, 
        act,
        rew, 
        next_obs, 
        terminated,
        truncated,
    ) -> dict:
        log_info = self._step(obs, act, rew, next_obs, terminated, truncated)
        self.step_idx += 1
        return log_info

    def _step(self, 
        obs, 
        act,
        rew, 
        next_obs, 
        terminated,
        truncated,
    ) -> dict:
        raise NotImplementedError()
    
    def train(self) -> dict:
        raise NotImplementedError()

class ReplayBuffer:
    def __init__(self, capacity=1000000, seed=42):
        self.max_size = capacity
        self.rand_gen = np.random.RandomState(seed)
        self.n_inserts = 0
        self.last_batch = None
        self.observations = None
        self.actions = None
        self.rewards = None
        self.next_observations = None
        self.terminateds = None
        self.truncateds = None

    @property
    def size(self):
        return min(self.n_inserts, self.max_size)

    def __len__(self):
        return self.size

    def sample(self, batch_size) -> dict[str, np.ndarray]:
        '''
        Samples batch_size entries from the replay buffer.
        '''
        rand_indices = self.rand_gen.randint(0, self.size, size=batch_size)
        batch = {
            "observations": self.observations[rand_indices],
            "actions": self.actions[rand_indices],
            "rewards": self.rewards[rand_indices],
            "next_observations": self.next_observations[rand_indices],
            "terminateds": self.terminateds[rand_indices],
            "truncateds": self.truncateds[rand_indices],
        }
        self.last_batch = batch
        return batch

    def insert(
        self,
        obs,
        act,
        rew,
        next_obs,
        terminated,
        truncated,
    ):
        """
        Inserts a single transition into the replay buffer.
        """
        rew = np.array(rew)
        act = np.array(act, dtype=np.int64)
        terminated = np.array(terminated)
        truncated = np.array(truncated)

        # If this is the first insert, set up buffer with shape and dtypes from given data
        if self.observations is None:
            self.observations = np.empty(
                (self.max_size, *obs.shape), 
                dtype=obs.dtype)
            self.actions = np.empty(
                (self.max_size, *act.shape), 
                dtype=act.dtype)
            self.rewards = np.empty(
                (self.max_size, *rew.shape), 
                dtype=rew.dtype)
            self.next_observations = np.empty(
                (self.max_size, *next_obs.shape), 
                dtype=next_obs.dtype)
            self.terminateds = np.empty(
                (self.max_size, *terminated.shape), 
                dtype=terminated.dtype)
            self.truncateds = np.empty(
                (self.max_size, *truncated.shape), 
                dtype=truncated.dtype)

        # Check to make sure new data size matches prev stored data
        assert obs.shape == self.observations.shape[1:]
        assert act.shape == self.actions.shape[1:]
        assert rew.shape == ()
        assert next_obs.shape == self.next_observations.shape[1:]
        assert terminated.shape == ()
        assert truncated.shape == ()

        # self.n_inserts % self.max_size gives the idx of the curr position to be overwritten
        idx = self.n_inserts % self.max_size
        self.observations[idx] = obs
        self.actions[idx] = act
        self.rewards[idx] = rew
        self.next_observations[idx] = next_obs
        self.terminateds[idx] = terminated
        self.truncateds[idx] = truncated

        self.n_inserts += 1