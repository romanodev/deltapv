from sesame_np_junction import sesame_pnj
from jaxpv_np_junction import jaxpv_pnj
import matplotlib.pyplot as plt
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save')
    parser.add_argument('--G')
    args = parser.parse_args()
    
    G = float(args.G) if args.G else None
    
    vj, jj = jaxpv_pnj(G=G)
    vs, js = sesame_pnj(G=G, voltages=vj)
    
    plt.clf()
    plt.plot(vs, js, label='sesame', marker='.')
    plt.plot(vj, jj, label='jaxpv', marker='.')

    plt.legend()
    if args.save is not None:
        plt.savefig(args.save)
    plt.show()