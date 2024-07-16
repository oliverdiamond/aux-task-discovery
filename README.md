# aux-task-discovery
Reproducing results from [Auxiliary Task Discovery Through Generate-And-Test](https://arxiv.org/abs/2210.14361) by Rafiee et al.

## Installation
1. Set up virtual env with python 3.9
```
conda create -n aux-task-discovery python=3.9
```
2. Install requirements
```
pip install -r requirements.txt
```
3. Install aux_task_discovery package in editable mode
```
pip install -e .
```

## Running Experiments
### To execute an individual run:

```
python3 aux_task_discovery/experiments/run_experiment.py
```
Arguments for running experiments can be found [here](./aux_task_discovery/experiments/argument_handling.py)

Note that arguments for the agent constructor are all specified as strings with the following format:
```
--agent_args arg1='arg1' arg2='0.1'
```

### To execute multiple runs or sweeps with wandb:

```
python3 aux_task_discovery/experiments/run_sweep.py --sweep_config='name_of_your_yaml_config'
```

All non-default arguments should be specified in a .yaml file stored at **./aux_task_discovery/experiments/sweep_configs/** with the format as shown [here](./aux_task_discovery/experiments/sweep_configs/dqn.yaml)
