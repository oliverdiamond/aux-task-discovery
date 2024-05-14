from aux_task_discovery.envs.gridworld import GridWorldEnv

class MazeEnv(GridWorldEnv):
    def __init__(self, seed=42):
        # Add boarder walls
        obstacles = [
            (x,y) 
            for x in range(11) 
            for y in range(11) 
            if (x in [0,10]) or (y in [0,10])
        ]
        # Add interior obstacles
        obstacles += [(2,2),(2,3),(2,4),(2,5),(2,6),
                      (2,7),(2,8),(3,2),(4,2),(5,2),
                      (6,2),(7,2),(8,2),(8,3),(8,4),
                      (8,5),(8,6),(8,7),(8,8),(3,8),
                      (4,8),(6,8),(7,8),(4,4),(4,5),
                      (4,6),(5,6),(6,4),(6,5),(6,6)]

        super().__init__(
            size=11, 
            start_pos=(9,1),
            goal_pos=(5,5),
            obstacles=obstacles,
            action_noise=0.5,
            seed=seed)
        

if __name__ == "__main__":
    # Print grid as string for reference
    env = MazeEnv()
    env.print_grid()