import os
import time
from spinup.utils.run_utils import setup_logger_kwargs
import numpy as np
# kwargs = setup_logger_kwargs('sac', '0')
# value = np.ones([2,3])

def record_network_value(value, output_dir=None, exp_name=None):
    # append DATE_TIME to dict
    array_name = output_dir + "/network_output.txt"
    # Save Dictionary to a csv
    with open(array_name, 'a') as f:
        np.savetxt(f, value, fmt='%2.1f')

# record_network_value(value, **kwargs)