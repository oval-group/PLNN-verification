#!/usr/bin/env python
import argparse
import torch

from plnn.model import load_rlv


def main():
    parser = argparse.ArgumentParser(description='Convert a .rlv file used as '
                                     'input for the PLANET verifier'
                                     'into a pytorch module')
    parser.add_argument('rlv_infile', type=argparse.FileType('r'),
                        help='.rlv file to convert.')
    parser.add_argument('pth_outfile', type=argparse.FileType('wb'),
                        help='Where to dump the resulting Pytorch module.')
    args = parser.parse_args()

    network_layers = load_rlv(args.rlv_infile)
    network = torch.nn.Sequential(*network_layers)
    torch.save(network, args.pth_outfile)


if __name__ == '__main__':
    main()
