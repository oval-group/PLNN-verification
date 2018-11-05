#!/usr/bin/env python
import argparse

from plnn.model import dump_rlv, load_rlv, simplify_network
from torch import nn


def main():
    parser = argparse.ArgumentParser(description="Read a .rlv file describing a net"
                                     "and create a new one to prove that the output"
                                     "is greater than a certain value. Rather than setting "
                                     "the value, you can simply setup by how much you want "
                                     "this to be true."
                                     "If this is negative, the property will be false."
                                     "If this is positive, the property will be true.")
    parser.add_argument('rlv_infile', type=argparse.FileType('r'),
                        help='.rlv file to change')
    parser.add_argument('min_file', type=argparse.FileType('r'),
                        help='File containing the value of the global minimum (or our best guess of it)')
    parser.add_argument('margin', type=float,
                        help='What should the margin compared to ')
    parser.add_argument('rlv_outfile', type=argparse.FileType('w'),
                        help='.rlv file to generate')
    args = parser.parse_args()

    net_layers, domain, prop_layers = load_rlv(args.rlv_infile)
    net_min = float(args.min_file.read())

    to_add = -net_min + args.margin

    bias_shift = nn.Linear(1, 1)
    bias_shift.weight.data.fill_(1)
    bias_shift.bias.data.fill_(to_add)
    prop_layers.append(bias_shift)

    all_layers = net_layers + prop_layers
    all_layers = simplify_network(all_layers)
    dump_rlv(args.rlv_outfile, all_layers, domain)


if __name__ == '__main__':
    main()
