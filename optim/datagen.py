import solver
import numpy as np

keys = ['NcS', 'NvS', 'EgS', 'epsilonS', 'EtS', 'mu_eS', 'mu_hS', 'tau_eS', 'tau_hS', 'affinityS', 'NcTe', 'NvTe',
        'EgTe', 'epsilonTe', 'EtTe', 'mu_eTe', 'mu_hTe', 'tau_eTe', 'tau_hTe', 'affinityTe', 'nD', 'nA']

log_rdict = {'NcS': (17, 19),
             'NvS': (19, 20),
             'EgS': (0, 1),
             'epsilonS': (0.5, 1.5),
             'EtS': (-11, -10),
             'mu_eS': (2, 3),
             'mu_hS': (1, 2),
             'tau_eS': (-9, -7),
             'tau_hS': (-14, -8),
             'affinityS': (0, 1),
             'NcTe': (17, 19),
             'NvTe': (19, 20),
             'EgTe': (0, 1),
             'epsilonTe': (0.5, 1.5),
             'EtTe': (-11, -10),
             'mu_eTe': (2, 3),
             'mu_hTe': (1, 2),
             'tau_eTe': (-9, -7),
             'tau_hTe': (-14, -8),
             'affinityTe': (0, 1),
             'nD': (14, 18),
             'nA': (14, 18)}


def dicts(log_rdict):
    rdict = {key: tuple(10 ** item for item in log_rdict[key]) for key in log_rdict}
    means = {key: sum(rdict[key]) / 2 for key in rdict}
    sds = {key: rdict[key][1] - rdict[key][0] for key in rdict}
    return rdict, means, sds


def draw(bounds):
    min, max = bounds
    prop = min + (max - min) * np.random.rand()
    return prop


def sample(rdict):
    params = {key: draw(rdict[key]) for key in rdict}
    return params


def normalize(params, means, vars):
    return {key: 2 * (params[key] - means[key]) / vars[key] for key in params}


def generate(params, ses):
    CdS = {'Nc': params['NcS'], 'Nv': params['NvS'], 'Eg': params['EgS'], 'epsilon': params['epsilonS'],
           'Et': params['EtS'], 'mu_e': params['mu_eS'], 'mu_h': params['mu_hS'], 'tau_e': params['tau_eS'],
           'tau_h': params['tau_hS'], 'affinity': params['affinityS']}
    CdTe = {'Nc': params['NcTe'], 'Nv': params['NvTe'], 'Eg': params['EgTe'], 'epsilon': params['epsilonTe'],
            'Et': params['EtTe'], 'mu_e': params['mu_eTe'], 'mu_h': params['mu_hTe'], 'tau_e': params['tau_eTe'],
            'tau_h': params['tau_hTe'], 'affinity': params['affinityTe']}
    print(CdS)
    print(CdTe)

    nD = params['nD']
    nA = params['nA']

    """CdS = {'Nc': 2.2e18, 'Nv': 1.8e19, 'Eg': 2.4, 'epsilon': 10, 'Et': 0,
           'mu_e': 100, 'mu_h': 25, 'tau_e': 1e-8, 'tau_h': 1e-13,
           'affinity': 4.}
    CdTe = {'Nc': 8e17, 'Nv': 1.8e19, 'Eg': 1.5, 'epsilon': 9.4, 'Et': 0,
            'mu_e': 320, 'mu_h': 40, 'tau_e': 5e-9, 'tau_h': 5e-9,
            'affinity': 3.9}

    nD = 1e17  # donor density [cm^-3]
    nA = 1e15  # acceptor density [cm^-3]"""

    mp = ses.solve(CdS, CdTe, nD, nA)
    return mp


if __name__ == '__main__':
    ses = solver.default()
    rdict, means, sds = dicts(log_rdict)
    params = sample(rdict)
    print(generate(params, ses))
