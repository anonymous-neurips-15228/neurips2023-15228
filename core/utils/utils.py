import yaml
import torch
import random
import numpy as np


def load_yaml(filename):
    with open(filename, 'r') as stream:
        file = yaml.safe_load(stream)
        return file

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True