import os
import random

import torch
import numpy as np
import shutil

from config import lr_decay


def setup_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_checkpoint(state, directory, is_best, filename='checkpoint.pth'):
    """Saves checkpoint to disk"""
    # directory = "runs/%s/%s/%s/"%(config.dataset, config.model, config.checkname)

    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(directory, 'model_best.pth'))


def load_checkpoint_1(directory, filename='checkpoint.pth'):
    filename = os.path.join(directory, filename)
    state = None
    if os.path.exists(filename):
        state = torch.load(filename)
    return state


def exist_checkpoint(directory, filename='checkpoint.pth'):
    return os.path.exists(os.path.join(directory, filename))


def adjust_learning_rate(optimizer, epoch):
    lr = 1e-2 * (0.1 ** max(0, (epoch - 1) // lr_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
