from gymnasium.envs.registration import register

register(
     id="maze",
     entry_point="aux_task_discovery.envs:MazeEnv",
)
register(
     id="fourrooms",
     entry_point="aux_task_discovery.envs:FourRoomsEnv",
)
register(
     id="gridworld",
     entry_point="aux_task_discovery.envs:GridWorldEnv",
)
