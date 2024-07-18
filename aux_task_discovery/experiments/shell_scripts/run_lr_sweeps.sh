#!/bin/bash

python3 aux_task_discovery/experiments/run_sweep.py --sweep_config gen_test
python3 aux_task_discovery/experiments/run_sweep.py --sweep_config dqn
python3 aux_task_discovery/experiments/run_sweep.py --sweep_config corner
python3 aux_task_discovery/experiments/run_sweep.py --sweep_config hallway