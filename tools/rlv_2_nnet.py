#!/usr/bin/env python
import argparse
import torch

from plnn.model import load_rlv, dump_nnet
torch.set_default_tensor_type('torch.DoubleTensor')


def main():
    parser = argparse.ArgumentParser(description="Load up a .rlv file used as"
                                     "input for the PLANET verifier, convert "
                                     "it to a pytorch network, and dump it as "
                                     "a .nnet file, after having removed the "
                                     "Maxpooling units.")
    parser.add_argument('rlv_infile', type=argparse.FileType('r'),
                        help='.rlv file to transform.')
    parser.add_argument('nnet_outfile', type=argparse.FileType('w'),
                        help='Where to dump as a .nnet file')
    args = parser.parse_args()

    net_layers, domain, prop_layers = load_rlv(args.rlv_infile)
    all_layers = net_layers + prop_layers
    dump_nnet(args.nnet_outfile, all_layers, domain)

if __name__ == '__main__':
    main()
