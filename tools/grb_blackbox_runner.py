#!/usr/bin/env python
import argparse

from plnn.black_box import BlackBoxNetwork
from plnn.model import load_and_simplify
from torch.autograd import Variable


def main():
    parser = argparse.ArgumentParser(description="Read a .rlv file"
                                     "and prove its property.")

    parser.add_argument('rlv_infile', type=argparse.FileType('r'),
                        help='.rlv file to prove.')
    parser.add_argument('--use_obj_function', action='store_true')
    args = parser.parse_args()

    bb_network, domain = load_and_simplify(args.rlv_infile,
                                           BlackBoxNetwork)
    bb_network.setup_model(domain,
                           use_obj_function=args.use_obj_function)
    sat, solution, nb_visited_states = bb_network.solve(domain)

    if sat is False:
        print("UNSAT")
    else:
        # Verify that it is a valid solution
        candidate_ctx = solution[0].view(1, -1)
        val = bb_network.net(candidate_ctx)
        margin = val.squeeze().data.item()
        if margin > 0:
            print("Error")
        else:
            print("SAT")
        print(solution[0])
        print(margin)
    print(f"Nb states visited: {nb_visited_states}")

if __name__ == '__main__':
    main()
