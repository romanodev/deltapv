import psc
from scipy.optimize import minimize
import logging
logger = logging.getLogger("jaxpv")
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

    return psc.gradf(x)


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
"""
Trial 1: Bounds defined as (x0 - 1, x0 + 1)

     fun: DeviceArray(-18.7958861, dtype=float64)  [initial: -18.70141521674728]
     jac: array([-3.27836900e-03, -1.18197708e-01,  2.23579420e-04, -7.03588728e-03,
        2.55203008e-10, -3.35201107e-09, -1.19807938e-07,  9.68635720e-01,
        9.48487414e-01, -5.91245330e-05,  1.74004535e-08, -5.64600516e-02,
       -1.48022086e-06, -9.43856117e-10,  1.05967096e-02,  5.48512678e-02,
        5.40691571e-01, -5.41958954e-01])
 message: 'Optimization terminated successfully'
    nfev: 19
     nit: 3
    njev: 3
  status: 0
 success: True
       x: array([  4.00302825,   3.9       ,   8.40021479,  18.63595228,
        18.        , 191.40000001,   5.40000017,   3.31105319,
         2.08800667,  20.00004471,  19.29999999,  18.00611068,
         4.50000112, 361.        ,  17.96729121,  17.99514153,
         3.9       ,   5.39905986])
"""
"""
Failing:
[3.54206248721763, 3.9000000000007304, 1.3997120767289741, 17.000000000000004,
18.7517009928743, 215.58363591614068, 430.34070902972456, 3.315033569080857,
2.08496643091869, 3.250144940557996, 18.271562864172243, 17.555835185442415,
39.658400933897404, 170.04590415933018, 19.99999999999996, 19.999999999999996,
4.7441879244528575, 4.7441879244529686]
"""
"""
Trial 2: Wide bounds

Optimization terminated successfully    (Exit mode 0)
            Current function value: -18.77884178682181
            Iterations: 6
            Function evaluations: 19
            Gradient evaluations: 6
     fun: DeviceArray(-18.77884179, dtype=float64)
     jac: array([-1.98515529e-03, -3.55332294e-02,  5.28112400e-04, -2.11516568e-03,
        5.89694260e-10, -5.56097241e-11, -3.35738551e-11,  5.07936836e-02,
        4.11092405e-02, -2.82246698e-04,  3.74665891e-07, -2.44671164e-03,
        2.82333277e-07, -5.28372563e-08,  2.18293397e-03,  2.06133824e-03,
        9.79579424e-02, -1.03592068e-01])
 message: 'Optimization terminated successfully'
    nfev: 19
     nit: 6
    njev: 6
  status: 0
 success: True
       x: array([  3.9518632 ,   3.92177767,   1.39785952,  17.01335127,
        18.75170086, 215.58363592, 430.34070903,   3.19959275,
         2.00876491,   3.25084181,  18.27156182,  17.60708458,
        39.65840014, 170.04591322,  19.98740757,  19.99101872,
         3.92177767,   4.97098514])
"""
"""
Trial 3: Flat bands

Iteration limit reached    (Exit mode 9)
            Current function value: -29.236985976300424
            Iterations: 10
            Function evaluations: 39
            Gradient evaluations: 9
     fun: DeviceArray(-29.23698598, dtype=float64)
     jac: array([-1.17460905e-01,  5.61452238e-01,  5.02245898e-04,  3.34162048e-02,
        5.04224843e-06, -9.72523017e-07,  5.91291325e-12,  1.32131044e+02,
        1.31727724e+02,  2.68427329e-02,  5.03595722e-04, -7.84118858e+00,
        1.74619505e-05, -7.64425345e-13, -2.46909359e-03,  1.23172660e+01])
 message: 'Iteration limit reached'
    nfev: 39
     nit: 10
    njev: 9
  status: 9
 success: False
       x: array([  1.4344334 ,   4.88767481,   6.13460402,  19.99999209,
        17.00002294, 102.60237715, 473.07117574,   2.2258033 ,
         1.        ,   2.13719668,  19.40511096,  17.00000379,
       448.65494922, 311.63744303,  17.00000113,  19.11751152])
"""