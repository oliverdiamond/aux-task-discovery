from envs.gridworld import GridWorldEnv

class MazeEnv(GridWorldEnv):
    def __init__(self, seed):
        # Add boarder walls
        obstacles = [
            (x,y) 
            for x in range(11) 
            for y in range(11) 
            if (x in [0,10]) or (y in [0,10])
        ]
        # Add interior obstacles
        obstacles += 

        super().__init__(
            size=11, 
            start_pos=(9,1),
            goal_pos=(5,5),
            obstacles=obstacles,
            action_noise=0.5,
            seed=seed)