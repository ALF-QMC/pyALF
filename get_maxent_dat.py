#!/usr/bin/env python3

from argparse import ArgumentParser
import h5py
import numpy as np

def _get_arg_parser():
    parser = ArgumentParser(
        description='Get info for MaxEnt from HDF5 data file.',
        )
    parser.add_argument(
        '--datafile', default="./data.h5",
        help='Data file containing the bins.')
    parser.add_argument(
        '--obs_name', '-o', required=True,
        help='Observable name. E.g. "Green_tau"')
    return parser


if __name__ == '__main__':
    parser = _get_arg_parser()
    args = parser.parse_args()

    with h5py.File(args.datafile, "r") as f:
        dtau = f[args.obs_name].attrs['dtau']
        channel = f[args.obs_name].attrs['Channel'].decode()
        obs = f[args.obs_name + "/obser"]
        N_orb = obs.shape[1]
        N_tau = obs.shape[3]

        orbital_coords = np.array([
            f[args.obs_name + "/lattice"].attrs[f'Orbital{i+1}'] for i in range(N_orb)])

    print(channel)

    print(f'Channel={channel}, dtau={dtau},  N_tau={N_tau}')
    print('Orbital coordinates:')
    print(orbital_coords)

    with open('maxent_aux', 'w', encoding='UTF-8') as f:
        f.write(f'Channel={channel}, dtau={dtau},  N_tau={N_tau}\n')
