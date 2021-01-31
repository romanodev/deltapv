import psc
from scipy.optimize import shgo
import logging
logger = logging.getLogger("deltapv")
logger.addHandler(logging.FileHandler("logs/shgo_1hr.log"))


def func(x):
    logger.info(f"{list(x)} objective")
    return psc.f(x)


def dfunc(x):
    logger.info(f"{list(x)} gradient")
    return psc.df(x)


bounds = psc.bounds
constraints = ({
    "type": "ineq",
    "fun": psc.g1,
    "jac": psc.jac1
}, {
    "type": "ineq",
    "fun": psc.g2,
    "jac": psc.jac2
}, {
    "type": "ineq",
    "fun": psc.g3,
    "jac": psc.jac3
}, {
    "type": "ineq",
    "fun": psc.g4,
    "jac": psc.jac4
}, {
    "type": "ineq",
    "fun": psc.g5,
    "jac": psc.jac5
})

if __name__ == "__main__":

    result = shgo(
        func,
        bounds=bounds,
        constraints=constraints,
        callback=lambda xk: print(f"SHGO iteration ended with x = {list(xk)}"),
        minimizer_kwargs={"method": "SLSQP", "options": {"ftol": 1e-2, "disp": True}},
        options={
            "maxtime": 3600,
            "jac": dfunc,
            "f_min": -18.81,
            "f_tol": 1e-2,
            "infty_constraints": True,
            "sampling_method": "simplicial"
        })
