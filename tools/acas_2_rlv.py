#!/usr/bin/env python
import argparse
from plnn.model import AcasNetwork


def main():
    parser = argparse.ArgumentParser(description='Convert a .nnet file used as input for the Reluplex verifier'
                                     'into a .rlv file used as input for the PLANET verifier')

    parser.add_argument('rpx_infile', type=argparse.FileType('r'),
                        help='.nnet file to convert.')
    parser.add_argument('rlv_outfile', type=argparse.FileType('w'),
                        help='Where to write the resulting .rlv file.')
    parser.add_argument('--property', type=argparse.FileType('r'),
                        help='Property to append to the .rlv file.')

    args = parser.parse_args()

    # Write down the network
    network = AcasNetwork(args.rpx_infile)
    network.write_rlv_file(args.rlv_outfile)

    # Add the property to prove if there is one to prove
    if args.property is not None:
        for assert_line in args.property:
            args.rlv_outfile.write(assert_line)


if __name__ == '__main__':
    main()
