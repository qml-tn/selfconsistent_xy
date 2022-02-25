import argparse
from selfconsistent_xy.quadratic import single_trajectory_benettin_rescaling
import numpy as np

def phase_diagram_slice(eta, n, m, ginit, etainit, dt, ntim, savedir, output, gsinit, geval_start):
    glist = np.arange(gevol_start, gevol_end, 0.02)
    for g in glist:
        print(n, eta, g)
        params = (g, eta, n, m, ginit, etainit, dt,
                  ntim, savedir, output, gsinit)
        single_trajectory_benettin_rescaling(params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--eta',
                        type=float,
                        default=0,
                        help='Interaction angle.')
    parser.add_argument('--gevol_start',
                        type=float,
                        default=0,
                        help='ZZ interaction strength for the evolution Hamiltonian. Start of the sweep.')
    parser.add_argument('--gevol_end',
                        type=float,
                        default=10,
                        help='ZZ interaction strength for the evolution Hamiltonian. End of the sweep.')
    parser.add_argument('--n',
                        type=int,
                        default=100,
                        help='Size of the chain.')
    parser.add_argument('--m',
                        type=int,
                        default=0,
                        help='Number of the largest Lyapunov exponents to calculate.')
    parser.add_argument('--ginit',
                        type=float,
                        default=1e-4,
                        help='Initial transverse field.')
    parser.add_argument('--etainit',
                        type=float,
                        default=0.1,
                        help='Angle of the initial state.')
    parser.add_argument('--dt',
                        type=float,
                        default=0.1,
                        help='Output step.')
    parser.add_argument('--ntim',
                        type=int,
                        default=2000,
                        help='Output steps.')
    parser.add_argument('--savedir',
                        type=str,
                        required=True,
                        help='The full path to the folder where the results should be stored.')
    parser.add_argument('--output',
                        type=str,
                        default="save",
                        help='Output type: "save", "tlyap", "lyap". Default: "save"')
    parser.add_argument('--gsinit',
                        action='store_true',
                        help='If true the etainit will not be used. The initial state will be close to the ground state with current eta. Quench only in the "g" direction.')

    parse_args, unknown = parser.parse_known_args()

    if len(unknown) > 0:
        print('Unknown arguments: {}'.format(unknown))

    phase_diagram_slice(**parse_args.__dict__)
