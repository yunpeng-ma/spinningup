import os
import itertools
import numpy as np
from multiprocessing import Pool
"""
set hyper parameters range:
reference: https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
"""
# pi_lr = [1e-3, 6e-4, 3e-4, 1e-4, 5e-5]

clip_ratio = [0.01, 0.02, 0.1]
hidden = [256, 512, 1024]
pi_lr = [1e-4, 5e-5]
lam = [0.97, 1.0]
processes = list(itertools.product(clip_ratio, hidden, pi_lr, lam))
# print(processes)


def run_process(process):
    os.system('python ppo.py --clip_ratio={} --hid={} --pi_lr={} --lam={}'.format(process[0], process[1], process[2], process[3]))

pool = Pool(36)
pool.map(run_process, processes)
