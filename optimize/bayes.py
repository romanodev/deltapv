import psc
import numpy as np
import matplotlib.pyplot as plt
from skopt import gp_minimize
import logging
logger = logging.getLogger("deltapv")
logger.addHandler(logging.FileHandler("logs/bayes_new_lcb.log"))


def fun(x):
    logger.info(f"{list(x)}")
    return float(psc.f(x))


res = gp_minimize(
    fun,  # the function to minimize
    psc.bounds,  # the bounds on each dimension of x
    acq_func="EI",  # the acquisition function
    n_calls=100,  # the number of evaluations of f
    n_random_starts=20,  # the number of random initialization points
    noise=1e-5)  # the noise level (optional)

logger.info(res)
