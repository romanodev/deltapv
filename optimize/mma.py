import psc
import nlopt
import numpy as np
import logging
logger = logging.getLogger("jaxpv")
logger.addHandler(logging.FileHandler("logs/nlopt_mma_xinit.log"))


def objective(x, grad):
    logger.info(list(x))
    y, dy = psc.gradf(x)
    if grad.size > 0:
        grad[:] = dy
    return float(y)


def constraint(result, x, grad):
    result[:] = psc.g(x)
    Jg = psc.jac(x)
    if grad.size > 0:
        grad[:, :] = Jg


opt = nlopt.opt(nlopt.LD_MMA, psc.n_params)
opt.set_lower_bounds(list(psc.vl))
opt.set_upper_bounds(list(psc.vu))
opt.set_max_objective(objective)
opt.add_inequality_mconstraint(constraint, np.full(5, 1e-8))
opt.set_xtol_rel(1e-4)
opt.set_maxtime(1800)
x = opt.optimize(list(psc.x_init))
maxf = opt.last_optimum_value()

logger.info(f"optimum at {x}")
logger.info(f"minimum value = {maxf}")
logger.info(f"result code = {opt.last_optimize_result()}")
