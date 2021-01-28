import psc
import numpy as np
import matplotlib.pyplot as plt
from skopt import gp_minimize
import logging
logger = logging.getLogger("jaxpv")

res = gp_minimize(
    psc.f,  # the function to minimize
    psc.bounds,  # the bounds on each dimension of x
    acq_func="EI",  # the acquisition function
    n_calls=50,  # the number of evaluations of f
    n_random_starts=30,  # the number of random initialization points
    noise=1e-5)  # the noise level (optional)

print(res)
