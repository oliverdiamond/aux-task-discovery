import wandb

from aux_task_discovery.experiments.train import training_loop
from aux_task_discovery.experiments.argument_handling import make_and_parse_args
from aux_task_discovery.utils.constants import WANDB_PROJECT


if __name__ == '__main__':
    args = make_and_parse_args()

    wandb.login()
    training_loop(args=args)