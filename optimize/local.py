import os
os.environ["LOGLEVEL"] = "WARNING"
import psc
from scipy.optimize import minimize

bounds = list(zip(psc.x0 - 1, psc.x0 + 1))
cons = ({
    "type": "ineq",
    "fun": psc.g1,
    "jac": lambda _: psc.jac1
}, {
    "type": "ineq",
    "fun": psc.g2,
    "jac": lambda _: psc.jac2
}, {
    "type": "ineq",
    "fun": psc.g3,
    "jac": lambda _: psc.jac3
}, {
    "type": "ineq",
    "fun": psc.g4,
    "jac": lambda _: psc.jac4
})

results = minimize(psc.gradf,
                   psc.x0,
                   method="SLSQP",
                   jac=True,
                   bounds=bounds,
                   constraints=cons)

print(results)
