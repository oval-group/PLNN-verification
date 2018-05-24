#!/usr/bin/env python
import argparse

from plnn.best_guess_timeout import HeuristicNetwork
from plnn.model import load_and_simplify


def main():
    parser = argparse.ArgumentParser(description="Read a .rlv file"
                                     "and prove its property.")
    parser.add_argument('rlv_infile', type=argparse.FileType('r'),
                        help='.rlv file to prove')
    parser.add_argument('timeout', type=int,
                        help="Time (in seconds) before timing out.")
    parser.add_argument('noprogress_timeout', type=int,
                        help="Time (in seconds) without progress before timing out")
    parser.add_argument('--use_cuda', action='store_true')

    args = parser.parse_args()


    network, domain = load_and_simplify(args.rlv_infile,
                                        HeuristicNetwork)

    sat, (sol_inp, sol_val), nb_samples = network.guess_lower_bound(domain,
                                                                    args.timeout,
                                                                    args.noprogress_timeout,
                                                                    early_stop=True,
                                                                    use_cuda=args.use_cuda)

    if sat:
        print("SAT")
        print(sol_inp)
        print(sol_val)
    else:
        print("UNSAT")
        print(sol_inp)
        print(sol_val)

    print(f"Nb samples: {nb_samples}")
if __name__ == '__main__':
    main()
