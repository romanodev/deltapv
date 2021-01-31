import psc
from scipy.optimize import basinhopping
import logging
logger = logging.getLogger("deltapv")
logger.addHandler(logging.FileHandler("logs/basin.log"))


def func(x):
    logger.info(f"{list(x)}")
    return psc.vagf(x)


def callback(x, f, accept):
    logger.info(f"Minimum with value {f} found at x = {list(x)}")


if __name__ == "__main__":

    result = basinhopping(func,
                          x0=psc.x_init,
                          niter=10,
                          T=5,
                          minimizer_kwargs={
                              "method": "SLSQP",
                              "jac": True,
                              "options": {
                                  "ftol": 1e-2
                              }
                          },
                          disp=True,
                          callback=callback)
    logger.info(
        f"Finished with objective {result.fun} at x = {list(result.x)}")
    print(result)
