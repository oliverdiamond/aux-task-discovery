from aux_task_discovery.envs.gridworld import GridWorldEnv

class FourRoomsEnv(GridWorldEnv):
    def __init__(self, seed=42):
        # Add obstacles
        obstacles = [(0,3), 
                     (1,3), 
                     (2,3), 
                     (3,0), 
                     (3,2), 
                     (3,3), 
                     (4,3), 
                     (4,4), 
                     (4,6), 
                     (6,3)]
        super().__init__(
            size=7, 
            start_pos=(6,0),
            goal_pos=(0,5),
            obstacles=obstacles,
            action_noise=0.5,
            seed=seed)
        
if __name__ == "__main__":
    # Print grid as string for reference
    env = FourRoomsEnv()
    env.print_grid()