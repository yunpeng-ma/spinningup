import os
import itertools
from multiprocessing import Pool
"""
set hyper parameters range:
reference: https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
"""
pi_lr = [1e-3, 6e-4, 3e-4, 1e-4, 5e-5]
hidden = [32, 64, 128]

processes = list(itertools.product(pi_lr, hidden))
# gamma = [0.8, 0.9, 0.99]
# lam = [0.9, 0.95, 0.97, 0.99]
# clip_ratio = [0.1, 0.2, 0.3]

def run_process(process):
    os.system('python ppo.py --pi_lr={} --hid={}'.format(process[0],process[1]))

pool = Pool(12)
pool.map(run_process, processes)
