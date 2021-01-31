import psc
from scipy.optimize import minimize
import logging
logger = logging.getLogger("deltapv")
logger.addHandler(logging.FileHandler("slsqp.log"))

cons = ({
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


def fun(x):

    logger.info(list(x))

    return psc.vagf(x)


if __name__ == "__main__":
    results = minimize(fun,
                       psc.x_init,
                       method="SLSQP",
                       jac=True,
                       options={
                           "disp": True,
                           "maxiter": 10
                       },
                       bounds=psc.bounds,
                       constraints=cons)

    print(results)
