import os
import itertools
import numpy as np
from multiprocessing import Pool
"""
set hyper parameters range:
reference: https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
"""
# pi_lr = [1e-3, 6e-4, 3e-4, 1e-4, 5e-5]

clip_ratio = [0.05, 0.1, 0.2]
hidden = [128, 256]
pi_lr = [3e-4, 1e-4]
processes = list(itertools.product(clip_ratio, hidden, pi_lr))
# print(processes)


def run_process(process):
    os.system('python ppo.py --clip_ratio={} --hid={} --pi_lr={}'.format(process[0], process[1], process[2]))


pool = Pool(12)
pool.map(run_process, processes)
