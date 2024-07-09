import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
matplotlib.use('agg')

from aux_task_discovery.envs import GridWorldEnv, FourRoomsEnv

def plot_subgoals(subgoals: np.array, gridworld_env: GridWorldEnv):
    '''
    Plots gridworld with specified subgoals marked as green squares
    '''
    # Set the non-zero element in each subgoal to the id for the aux task
    # This allows for identifying which aux task each plotted subgoal corresponds to
    for i in range(len(subgoals)):
        # Find the index of the non-zero element
        non_zero_index = np.argmax(subgoals[i])
        # Set the non-zero element to the 
        subgoals[i][non_zero_index] = i+1

    subgoals = subgoals.sum(axis=0).reshape(gridworld_env.unwrapped.size, gridworld_env.unwrapped.size) # 1 if cell is a subgoal else 0
    dim = gridworld_env.unwrapped.size
    fig, ax = plt.subplots(dpi=300)
    for i in range(dim+2):
        for j in range(dim+2):
            # Set subgoals to green, walls and obstacles to grey
            if i == 0 or i == dim + 1 or j == 0 or j == dim + 1: # Wall
                color = 'dimgrey'
            else:
                # Adjust the indices to match the actual grid cells
                grid_i = i - 1
                grid_j = j - 1
                if gridworld_env.unwrapped.is_obstacle((grid_i,grid_j)): # Obstacle
                    color = 'dimgrey'
                elif subgoals[grid_i,grid_j]!=0: # Subgoal
                    color = 'lime'
                    # Add text for aux task index
                    ax.text(j+0.5, dim + 1 - i + 0.2, f'aux_{int(subgoals[grid_i,grid_j])}', fontsize=10, fontweight='bold', fontname='Arial', ha='center', va='center')
                else:
                    color = 'white'
                # Add text for start position
                if (grid_i,grid_j) == gridworld_env.unwrapped.start_pos:
                    ax.text(j+0.5, dim + 1 - i + 0.5, 'S', fontsize=16, fontweight='bold', fontname='Arial', ha='center', va='center')
                # Add text for goal position
                if (grid_i,grid_j) == gridworld_env.unwrapped.goal_pos:
                    ax.text(j+0.5, dim + 1 - i + 0.5, 'G', fontsize=16, fontweight='bold', fontname='Arial', ha='center', va='center')
            # Create a rectangle
            rect = patches.Rectangle((j, dim + 1 - i), 1, 1, facecolor=color, edgecolor='black', linewidth=1.25)
            # Add the rectangle to the plot
            ax.add_patch(rect)

    # Set the limits and aspect ratio
    ax.set_xlim(0, dim+2)
    ax.set_ylim(0, dim+2)
    ax.set_aspect('equal')

    # Remove the axes
    ax.axis('off')

    plt.close()

    return fig

#---------------TESTS---------------#

def test_plot_subgoals():
    env = FourRoomsEnv()
    subgoals = np.zeros((2, env.size*env.size), dtype=np.float32)
    subgoals[0, 0] = 1
    subgoals[1, 2] = 1
    fig = plot_subgoals(subgoals, env)
    plt.show()


if __name__ == "__main__":
    test_plot_subgoals()