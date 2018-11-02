#!/usr/bin/env python
import argparse

from plnn.mip_solver import MIPNetwork
from plnn.model import load_and_simplify
from torch.autograd import Variable

def main():
    parser = argparse.ArgumentParser(description="Read a .rlv file"
                                     "and prove its property.")

    parser.add_argument('rlv_infile', type=argparse.FileType('r'),
                        help='.rlv file to prove.')
    parser.add_argument('--sym_bounds', action='store_true')
    parser.add_argument('--use_obj_function', action='store_true')
    parser.add_argument('--interval-analysis', action='store_true')
    parser.add_argument('--paramfile', type=str,
                        help="Path to a parameter file to use")
    args = parser.parse_args()

    mip_network, domain = load_and_simplify(args.rlv_infile,
                                            MIPNetwork)
    mip_network.setup_model(domain,
                            sym_bounds=args.sym_bounds,
                            use_obj_function=args.use_obj_function,
                            interval_analysis=args.interval_analysis,
                            parameter_file=args.paramfile)

    sat, solution, nb_visited_states = mip_network.solve(domain)

    if sat is False:
        print("UNSAT")
    else:
        # Verify that it is a valid solution
        candidate_ctx = Variable(solution[0].view(1,-1))
        val = mip_network.net(candidate_ctx)
        margin = val.squeeze().item()
        if margin > 0:
            print("Error")
        else:
            print("SAT")
        print(solution[0])
        print(margin)
    print(f"Nb states visited: {nb_visited_states}")

if __name__ == '__main__':
    main()
