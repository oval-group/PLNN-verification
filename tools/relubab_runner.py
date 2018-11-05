#!/usr/bin/env python
import argparse

from plnn.relu_branch_and_bound import relu_bab
from plnn.network_linear_approximation import AssumptionLinearizedNetwork
from plnn.model import load_and_simplify

def main():
    parser = argparse.ArgumentParser(description="Read a .rlv file"
                                     "and prove its property.")
    parser.add_argument('rlv_infile', type=argparse.FileType('r'),
                        help='.rlv file to prove.')
    args = parser.parse_args()

    network, domain = load_and_simplify(args.rlv_infile,
                                        AssumptionLinearizedNetwork)

    epsilon = 0
    decision_bound = 0
    min_lb, min_ub, ub_point, nb_visited_states = relu_bab(network, domain,
                                                           epsilon, decision_bound)

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
