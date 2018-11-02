#!/usr/bin/env python
import argparse

from plnn.branch_and_bound import bab
from plnn.network_linear_approximation import LinearizedNetwork
from plnn.dual_network_linear_approximation import LooseDualNetworkApproximation
from plnn.model import load_and_simplify

def main():
    parser = argparse.ArgumentParser(description="Read a .rlv file"
                                     "and prove its property.")
    parser.add_argument('rlv_infile', type=argparse.FileType('r'),
                        help='.rlv file to prove.')
    parser.add_argument('--reluify_maxpools', action='store_true')
    parser.add_argument('--smart_branching', action='store_true')
    args = parser.parse_args()

    network, domain = load_and_simplify(args.rlv_infile,
                                        LinearizedNetwork)
    if args.reluify_maxpools:
        network.remove_maxpools(domain)

    if args.smart_branching:
        # Re-read the file to load it into another network
        args.rlv_infile.seek(0)
        smart_brancher, _ = load_and_simplify(args.rlv_infile,
                                              LooseDualNetworkApproximation)
        # This
        smart_brancher.remove_maxpools(domain)
    else:
        smart_brancher = None

    epsilon = 1e-2
    decision_bound = 0
    min_lb, min_ub, ub_point, nb_visited_states = bab(network, domain,
                                                      epsilon, decision_bound,
                                                      smart_brancher)

    if min_lb >= 0:
        print("UNSAT")
    elif min_ub < 0:
        # Verify that it is a valid solution
        candidate_ctx = ub_point.view(1,-1)
        val = network.net(candidate_ctx)
        margin = val.squeeze().item()
        if margin > 0:
            print("Error")
        else:
            print("SAT")
        print(ub_point)
        print(margin)
    else:
        print("Unknown")
    print(f"Nb states visited: {nb_visited_states}")


if __name__ == '__main__':
    main()
