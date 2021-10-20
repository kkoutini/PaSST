import torch
import numpy as np
import random


def worker_init_fn(x):
    seed = (torch.initial_seed() + x * 1000) % 2 ** 31  # problem with nearly seeded randoms

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return
