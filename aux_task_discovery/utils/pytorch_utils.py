from typing import Union

import torch
import numpy as np

device = None

def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("Using CPU.")

def from_numpy(data: Union[np.ndarray, dict], **kwargs):
    if isinstance(data, dict):
        return {k: from_numpy(v) for k, v in data.items()}
    else:
        data = torch.from_numpy(data, **kwargs)
        if data.dtype == torch.float64:
            data = data.float()
        return data.to(device)


def to_numpy(tensor: Union[torch.Tensor, dict]):
    if isinstance(tensor, dict):
        return {k: to_numpy(v) for k, v in tensor.items()}
    else:
        return tensor.to("cpu").detach().numpy()