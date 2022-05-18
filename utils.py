import os
import sys
import yaml
import torch
import random
import logging
import argparse
import numpy as np


class AverageMeter(object):
    """
    Computes and stores the average and current value of metrics.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """
        String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    with open(os.path.join('config', parser.parse_args().config), 'r') as stream:
        cfg = yaml.load(stream, Loader=yaml.FullLoader)
    return cfg

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_logger(log_dir, log_name, level=logging.INFO):
    file = os.path.join(log_dir, log_name)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if os.path.exists(file):
        os.remove(file)
    logger = logging.getLogger(__name__)
    logger.setLevel(level=level)
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    # StreamHandler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level=logging.DEBUG)
    logger.addHandler(stream_handler)
    # FileHandler
    file_handler = logging.FileHandler(file)
    file_handler.setLevel(level=level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger