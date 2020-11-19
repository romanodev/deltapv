from sesame_np_junction import sesame_pnj
from jaxpv_np_junction import jaxpv_pnj
import matplotlib.pyplot as plt
import numpy as np
import argparse


def error(G):
    v, jj = jaxpv_pnj(G=G)
    _, js = sesame_pnj(G=G, voltages=v)
    mse = ((jj - js) ** 2).mean()
    msre = (((jj - js) / js) ** 2).mean()
    return mse, msre

def error_curve():
    Gs = np.power(10, np.linspace(1, 23, 20))
    mses = []
    msres = []
    for G in Gs:
        mse, msre = error(G)
        mses.append(mse)
        msres.append(msre)
    return Gs, mses, msres
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save')
    args = parser.parse_args()
    
    Gs, mses, msres = error_curve()
    
    for group in [('mse', mses), ('msre', msres)]:
        label, error = group
        plt.clf()
        plt.plot(Gs, error, label=label, marker='.')
        plt.xlabel('uniform G')
        plt.xscale('log')
        plt.ylabel(label)
        plt.yscale('log')

        plt.legend()
        if args.save is not None:
            name, form = args.save.split('.')
            plt.savefig(f'{name}_{label}.{form}')
        plt.show()