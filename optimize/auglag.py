import psc
import nlopt
import numpy as np
import logging
logger = logging.getLogger("deltapv")
logger.addHandler(logging.FileHandler("logs/nlopt_auglag.log"))


def objective(x, grad):
    if grad.size > 0:
        y, dy = psc.vagf(x)
        grad[:] = dy
    else:
        y = psc.f(x)
    return float(y)


def constraint(result, x, grad):
    result[:] = psc.g(x)
    Jg = psc.jac(x)
    if grad.size > 0:
        grad[:, :] = Jg


mma = nlopt.opt(nlopt.LD_MMA, psc.n_params)
mma.set_lower_bounds(list(psc.vl))
mma.set_upper_bounds(list(psc.vu))
mma.set_xtol_rel(1e-4)
mma.set_ftol_rel(1e-4)
mma.set_maxeval(10)

mlsl = nlopt.opt(nlopt.G_MLSL_LDS, psc.n_params)
mlsl.set_local_optimizer(mma)
mlsl.set_population(5)
mlsl.set_lower_bounds(list(psc.vl))
mlsl.set_upper_bounds(list(psc.vu))
mlsl.set_maxtime(3600)

auglag = nlopt.opt(nlopt.AUGLAG, psc.n_params)
auglag.set_local_optimizer(mlsl)
auglag.set_lower_bounds(list(psc.vl))
auglag.set_upper_bounds(list(psc.vu))
auglag.set_max_objective(objective)
auglag.add_inequality_mconstraint(constraint, np.full(5, 1e-8))
auglag.set_xtol_rel(1e-4)
auglag.set_maxtime(3600)

x = auglag.optimize(list(psc.x_init))
maxf = auglag.last_optimum_value()

logger.info(f"optimum at {x}")
logger.info(f"minimum value = {maxf}")
logger.info(f"result code = {auglag.last_optimize_result()}")
