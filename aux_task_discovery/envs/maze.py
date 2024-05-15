from aux_task_discovery.envs.gridworld import GridWorldEnv

class MazeEnv(GridWorldEnv):
    def __init__(self, seed=42):
        # Add obstacles
        obstacles = [(1,1), (1,2), (1,3), (1,4), (1,5), 
                     (1,6), (1,7), (2,1), (3,1), (4,1), 
                     (5,1), (6,1), (7,1), (7,2), (7,3), 
                     (7,4), (7,5), (7,6), (7,7), (2,7), 
                     (3,7), (5,7), (6,7), (3,3), (3,4), 
                     (3,5), (4,5), (5,3), (5,4), (5,5)]
        super().__init__(
            size=9, 
            start_pos=(8,0),
            goal_pos=(4,4),
            obstacles=obstacles,
            action_noise=0.5,
            seed=seed)
        

if __name__ == "__main__":
    # Print grid as string for reference
    env = MazeEnv()
    env.print_grid()