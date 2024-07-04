import argparse

def make_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gpu", 
                        action="store_true",
                        help='Wheather to use gpu for training')
    parser.add_argument("--gpu_id", 
                        default=0,
                        help='ID of GPU to use for training if --use_gpu is set')
    parser.add_argument('--max_steps', 
                        type=int, 
                        default=1000000000,
                        help='Max number of environment interactions for the expirement. If both max_steps and max_episodes are specified, the experiment will end when either condition is met.')
    parser.add_argument('--max_episodes', 
                        type=int, 
                        default=1000000000,
                        help='Max number of episodes for the expirement. If both max_steps and max_episodes are specified, the experiment will end when either condition is met.')
    parser.add_argument('--seed', 
                        type=int, 
                        default=42,
                        help='Experiment random seed.')
    parser.add_argument('--env', 
                        type=str, 
                        default='fourrooms',
                        help='Gymnasium environment id')
    parser.add_argument('--agent', 
                        type=str, 
                        default='dqn',
                        help='Agent id, see ./aux_task_discovery/agents/__init__.py for agent registry')
    parser.add_argument('--agent_args', 
                        type=str, 
                        metavar='KEY=VALUE', 
                        nargs='+', 
                        default={},
                        help='Kwarg arguments passed to agent constructor')
    return parser

# Source: https://gist.github.com/fralau/061a4f6c13251367ef1d9a9a99fb3e8d
def parse_var(s):
    """
    Parse a key, value pair, separated by '='
    On the command line (argparse) a declaration will look like: 
        foo=hello
    Key will be treated as a string. Value will be evaulated, but can also be a string.
    """
    print(s)
    items = s.split('=', 1)
    key = items[0].strip()
    value = items[1]
    if value[0].isalpha():
        # Value is a string, add quotes so eval returns the string
        value = '\'' + value + '\''
    return (key, eval(value))

def parse_vars(items):
    """
    Parse a series of key-value pairs and return a dictionary
    """
    d = {}
    if items:
        for item in items:
            key, value = parse_var(item)
            d[key] = value
    return d

DICT_ARGS_LIST = ['agent_args']
    
def format_args(args):
    for arg_name in DICT_ARGS_LIST:
        if len(getattr(args, arg_name)) > 0:
            setattr(args, arg_name, parse_vars(getattr(args, arg_name)))
    return args

def make_and_parse_args():
    parser = make_arg_parser()
    args = parser.parse_args()
    args = format_args(args)
    return args