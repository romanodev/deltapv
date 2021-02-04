import psc
import nlopt
import numpy as np
import logging
logger = logging.getLogger("deltapv")
logger.addHandler(logging.FileHandler("logs/nlopt_isres.log"))


def objective(x, grad):
    logger.info(f"{list(x)}")
    logger.info(f"Complies with lower bounds: {np.alltrue(x >= psc.vl)}")
    logger.info(f"Complies with upper bounds: {np.alltrue(x <= psc.vu)}")
    logger.info(f"Complies with constraints: {np.alltrue(psc.g(x) <= 0)}")
    if grad.size > 0:
        y, dy = psc.vagf(x)
        grad[:] = dy
    else:
        y = psc.f(x)
    logger.info(f"Objective: {float(y)}")
    return float(y)


def constraint(result, x, grad):
    result[:] = psc.g(x)
    Jg = psc.jac(x)
    if grad.size > 0:
        grad[:, :] = Jg


isres = nlopt.opt(nlopt.GN_ISRES, psc.n_params)
isres.set_lower_bounds(list(psc.vl))
isres.set_upper_bounds(list(psc.vu))
isres.set_min_objective(objective)
isres.add_inequality_mconstraint(constraint, np.zeros(5))
isres.set_xtol_rel(1e-4)
isres.set_maxtime(900)

x = isres.optimize(list(psc.x_init))
maxf = isres.last_optimum_value()

logger.info(f"optimum at {x}")
logger.info(f"minimum value = {maxf}")
logger.info(f"result code = {isres.last_optimize_result()}")
