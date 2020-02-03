import scipy.constants as sc
import math

USE_JAX = False

def scales():
    T = 300
    KB = sc.k
    q , _ , _ = sc.physical_constants['elementary charge']
    eps_0 = sc.epsilon_0

    scales = {}

    scales['n'] = 1e19 #cm-3
    scales['t'] = eps_0 * 1e-2 / q / scales['n']
    scales['E'] = KB * T / q
    scales['d'] = math.sqrt( eps_0 * 1e-2 * scales['E'] / ( q * scales['n'] ) )
    scales['v'] = scales['d'] / scales['t']
    scales['J'] = ( KB * T * scales['n'] ) / scales['d']
    scales['U'] = scales['n'] * scales['E'] / ( scales['d']**2 )

    return scales
