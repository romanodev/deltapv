import scipy.constants as const
import math

##### Units required in inputs #####
## grid : cm
## eps : unitless
## Chi , Eg , Et : eV
## Nc , Nv , Ndop : cm^(-3)
## mn , mp : cm^2 / V / s
## tn , tp : s
## Snl , Snr , Spl , Spr : cm / s
## G : cm^(-3) / s


def scales():
    """
    Defines the scaling parameters for dimensional variables.

    Returns
    -------
        dictionnary ( 7 keys )
            Scaling parameters for dimensional variables.
            'd' -> scaling coefficient for the grid ( cm )
            'E' -> scaling coefficient for energies ( eV = J / C = V )
            'n' -> scaling coefficient for densities ( cm^(-3) )
            'J' -> scaling coefficient for current densities ( A / cm^2 )
            'm' -> scaling coefficient for mobilities ( cm^2 / V / s )
            'U' -> scaling coefficient for generation rates ( cm^(-3) / s )
            't' -> scaling coefficient for e-/hole lifetimes ( s )
            'v' -> scaling coefficient for recombination velocities ( cm / s )

    """
    T = 300
    KB = const.k
    q, _, _ = const.physical_constants['elementary charge']
    eps_0 = const.epsilon_0 * 1e-2  # C * V-1 * m-1 -> C * V-1 * cm-1

    scales = {}

    scales['n'] = 1e19
    scales['m'] = 1
    scales['E'] = KB * T / q
    scales['t'] = eps_0 / q / scales['n'] / scales['m']
    scales['d'] = math.sqrt(eps_0 * scales['E'] / (q * scales['n']))
    scales['v'] = scales['d'] / scales['t']
    scales['J'] = KB * T * scales['n'] * scales['m'] / scales['d']
    scales['U'] = scales['n'] * scales['E'] * scales['m'] / (scales['d']**2)
    return scales
