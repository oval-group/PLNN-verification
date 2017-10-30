#!/usr/bin/env python
import argparse

from plnn.branch_and_bound import bab
from plnn.network_linear_approximation import LinearizedNetwork
from plnn.model import load_and_simplify


def main():
    parser = argparse.ArgumentParser(description="Read a .rlv file"
                                     "and prove its property.")
    parser.add_argument('rlv_infile', type=argparse.FileType('r'),
                        help='.rlv file to prove.')
    args = parser.parse_args()

    network, domain = load_and_simplify(args.rlv_infile,
                                        LinearizedNetwork)

    epsilon = 1e-2
    decision_bound = 0
    min_lb, min_ub, ub_point = bab(network, domain,
                                   epsilon, decision_bound)

    if min_lb >= 0:
        print("UNSAT")
    elif min_ub < 0:
        print("SAT")
        print(ub_point)
    else:
        print("Unknown")


if __name__ == '__main__':
    main()
