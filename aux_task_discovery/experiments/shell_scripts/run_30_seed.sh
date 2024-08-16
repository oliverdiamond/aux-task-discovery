#!/bin/bash

python3 ./aux_task_discovery/experiments/run_sweep.py --config dqn_30_seed
python3 ./aux_task_discovery/experiments/run_sweep.py --config gentest_30_seed
python3 ./aux_task_discovery/experiments/run_sweep.py --config fixed_random_30_seed