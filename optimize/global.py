import psc
from scipy.optimize import shgo

bounds = list(zip(psc.x0 - 1, psc.x0 + 1))
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
})

result = shgo(psc.gradh,
              bounds=bounds,
              constraints=cons,
              minimizer_kwargs={"method": "SLSQP"},
              options={"jac": True})

print(result.x, result.fun)
