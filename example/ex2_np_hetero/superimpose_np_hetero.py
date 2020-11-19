from sesame_np_hetero import sesame_pn_hetero
from jaxpv_np_hetero import jaxpv_pn_hetero
import matplotlib.pyplot as plt
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save')
    parser.add_argument('--G')
    args = parser.parse_args()
    
    G = float(args.G) if args.G else None
    
    vj, jj = jaxpv_pn_hetero(G=G)
    vs, js = sesame_pn_hetero(G=G, voltages=vj)
    
    plt.clf()
    plt.plot(vs, js, label='sesame', marker='.')
    plt.plot(vj, jj, label='jaxpv', marker='.')

    plt.legend()
    if args.save is not None:
        plt.savefig(args.save)
    plt.show()