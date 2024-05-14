from gymnasium.envs.registration import register

register(
     id="envs/maze",
     entry_point="aux_task_discovery.envs:MazeEnv",
)
register(
     id="envs/fourrooms",
     entry_point="aux_task_discovery.envs:FourRoomsEnv",
)
register(
     id="envs/gridworld",
     entry_point="aux_task_discovery.envs:GridWorldEnv",
)
