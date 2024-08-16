#!/bin/bash

python3 ./aux_task_discovery/experiments/run_sweep.py --config dqn_lr
python3 ./aux_task_discovery/experiments/run_sweep.py --config gentest_lr
python3 ./aux_task_discovery/experiments/run_sweep.py --config fixed_random_lr