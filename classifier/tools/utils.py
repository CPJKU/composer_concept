""" Adaptation from original repo (https://github.com/KimSSung/Deep-Composer-Classification/blob/master/main.py) """

import torch
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)  # python random module
    np.random.seed(seed)  # np module
    torch.manual_seed(seed)  # for both CPU & GPU
