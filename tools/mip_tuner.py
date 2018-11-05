#!/usr/bin/env python
import argparse

from plnn.mip_solver import MIPNetwork
from plnn.model import load_and_simplify


def main():
    parser = argparse.ArgumentParser(description="Read a .rlv file"
                                     "and generate a tuned parameter file.")

    parser.add_argument('rlv_infile', type=argparse.FileType('r'),
                        help='.rlv file to prove.')
    parser.add_argument('tune_outfile', type=str,
                        help='Where te write the tuned parameters.'
                        'Needs to end in .prm')
    parser.add_argument('tune_timeout', type=int,
                        help="How much time to spend tuning the parameters (in s).")
    args = parser.parse_args()
    assert args.tune_outfile.endswith('.prm')

    mip_network, domain = load_and_simplify(args.rlv_infile,
                                            MIPNetwork)
    mip_network.setup_model(domain)

    mip_network.tune(args.tune_outfile, args.tune_timeout)


if __name__ == '__main__':
    main()
