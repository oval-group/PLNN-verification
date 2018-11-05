#!/usr/bin/env python
import argparse
import torch

from plnn.model import load_rlv, dump_rlv

# For the comparison with the rewrites, use double precision for the checking
torch.set_default_dtype(torch.float64)


def main():
    parser = argparse.ArgumentParser(description="Load up a .rlv file used as "
                                     "input for the PLANET verifier, convert"
                                     "it to a pytorch network, and re-dump it"
                                     "as a .rlv file, after having removed the"
                                     " Maxpooling units.")
    parser.add_argument('rlv_infile', type=argparse.FileType('r'),
                        help='.rlv file to transform.')
    parser.add_argument('rlv_outfile', type=argparse.FileType('w'),
                        help='Where to dump the new version of the rlv file')
    args = parser.parse_args()

    net_layers, domain, prop_layers = load_rlv(args.rlv_infile)
    all_layers = net_layers + prop_layers
    dump_rlv(args.rlv_outfile, all_layers, domain, True)


if __name__ == '__main__':
    main()
