import yaml
import os
import argparse

import wandb

from aux_task_discovery.utils import load_yaml_config
from aux_task_discovery.utils.constants import WANDB_PROJECT
from aux_task_discovery.experiments.train import training_loop

def main():
    # Parse Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_config", type=str, required=True)
    args = parser.parse_args()
    
    # Load sweep config from name of specified yaml file
    path = os.path.dirname(os.path.abspath(__file__)) + '/sweep_configs/' + args.sweep_config + '.yaml'
    sweep_config = load_yaml_config(path)

    # Start sweep
    wandb.login()
    sweep_id = wandb.sweep(sweep_config, project=WANDB_PROJECT)
    wandb.agent(sweep_id, training_loop)

if __name__ == '__main__':
    main()