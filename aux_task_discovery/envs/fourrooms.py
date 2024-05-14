from aux_task_discovery.envs.gridworld import GridWorldEnv

class FourRoomsEnv(GridWorldEnv):
    def __init__(self, seed=42):
        # Add boarder walls
        obstacles = [
            (x,y) 
            for x in range(9) 
            for y in range(9) 
            if (x in [0,8]) or (y in [0,8])
        ]
        # Add interior obstacles
        obstacles += [(1,4),
                      (2,4),
                      (3,4),
                      (4,1),
                      (4,3),
                      (4,4),
                      (5,4),
                      (5,5),
                      (5,7),
                      (7,4)]

        super().__init__(
            size=9, 
            start_pos=(7,1),
            goal_pos=(1,6),
            obstacles=obstacles,
            action_noise=0.5,
            seed=seed)
        
if __name__ == "__main__":
    # Print grid as string for reference
    env = FourRoomsEnv()
    env.print_grid()