import argparse
import os
import json

import numpy as np
import wandb
import gymnasium as gym
import pandas as pd

from aux_task_discovery.utils.constants import WANDB_PROJECT, WANDB_ENTITY

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", 
                        type=str, 
                        required=True,
                        help='wandb ID for the run or sweep to import data from')
    parser.add_argument("--sweep", 
                        action="store_true", 
                        help='Use if the provided ID is for a sweep, otherwise the ID is treated as an individual run by default')
    args = parser.parse_args()

    api = wandb.Api()

    if args.sweep:
        # Query W&B API
        sweep = api.sweep(WANDB_ENTITY + "/" + WANDB_PROJECT + "/" + args.id)
        # Make directories for sweep data
        sweep_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"sweep_data/{sweep.id}")
        data_dir = os.path.join(sweep_dir, 'data')
        summary_dir = os.path.join(sweep_dir, 'summary')
        config_dir = os.path.join(sweep_dir, 'config')
        os.makedirs(sweep_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(summary_dir, exist_ok=True)
        os.makedirs(config_dir, exist_ok=True)
        # Save sweep config
        with open(os.path.join(sweep_dir, "sweep_config.json"), "w") as f:
            json.dump(sweep.config, f, indent=4)
        # Save run config, summary, data
        for run in sweep.runs:
            with open(os.path.join(config_dir, f"{run.id}.json"), "w") as f:
                run.config['run_id'] = run.id
                run.config['run_name'] = run.name
                json.dump(run.config, f, indent=4)
            with open(os.path.join(summary_dir, f"{run.id}.json"), "w") as f:
                # Remove wandb internal metrics
                summary = {
                    key: val 
                    for key, val in dict(run.summary).items() 
                    if not key.startswith('_') 
                    and not isinstance(val, wandb.old.summary.SummarySubDict)
                }
                summary['run_id'] = run.id
                summary['run_name'] = run.name
                json.dump(summary, f, indent=4)
            run_data = pd.DataFrame([row for row in run.scan_history()])
            run_data.to_csv(os.path.join(data_dir, f"{run.id}.csv"), index=False)
    else:
        # Query W&B API
        run = api.run(WANDB_ENTITY + "/" + WANDB_PROJECT + "/" + args.id)
        # Make directory for run data
        run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"run_data/{run.id}")
        os.makedirs(run_dir, exist_ok=True)
        # Save run config, summary, data
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            run.config['run_id'] = run.id
            run.config['run_name'] = run.name
            json.dump(run.config, f, indent=4)
        with open(os.path.join(run_dir, "summary.json"), "w") as f:
            # Remove wandb internal metrics
            summary = {
                key: val 
                for key, val in dict(run.summary).items() 
                if not key.startswith('_') 
                and not isinstance(val, wandb.old.summary.SummarySubDict)
            }
            summary['run_id'] = run.id
            summary['run_name'] = run.name
            json.dump(summary, f, indent=4)
        run_data = pd.DataFrame([row for row in run.scan_history()])
        run_data.to_csv(os.path.join(run_dir, "data.csv"), index=False)


if __name__ == '__main__':
    main()