#!/usr/bin/env python
import argparse

from plnn.mip_solver import MIPNetwork
from plnn.model import load_and_simplify


def main():
    parser = argparse.ArgumentParser(description="Read a .rlv file"
                                     "and prove its property.")

    parser.add_argument('rlv_infile', type=argparse.FileType('r'),
                        help='.rlv file to prove.')
    args = parser.parse_args()

    mip_network, domain = load_and_simplify(args.rlv_infile,
                                            MIPNetwork)
    sat, solution = mip_network.solve(domain)

    if sat is False:
        print("UNSAT")
    else:
        print("SAT")
        print(solution)

if __name__ == '__main__':
    main()
