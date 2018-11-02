#!/usr/bin/env python
import argparse
import torch

from plnn.model import dump_nnet, dump_rlv, simplify_network
from torch import nn

torch.set_default_tensor_type('torch.DoubleTensor')
error_tol = 1e-5

def assert_network_greater_than(net, domain, min_theoretical):

    nb_samples = 1024 * 1024
    nb_inp = domain.size(0)
    rand_samples = torch.Tensor(nb_samples, nb_inp)
    rand_samples.uniform_(0, 1)

    domain_lb = domain.select(1, 0).contiguous()
    domain_ub = domain.select(1, 1).contiguous()
    domain_width = domain_ub - domain_lb

    domain_lb = domain_lb.view(1, nb_inp).expand(nb_samples, nb_inp)
    domain_width = domain_width.view(1, nb_inp).expand(nb_samples, nb_inp)

    inps = domain_lb + domain_width * rand_samples

    outs = net(inps)

    min_out = outs.min()
    assert min_out > min_theoretical - error_tol, "Ladder network not correct"

def generate_network(ladder_dimension, margin):
    '''
    Create a twin ladder network
    A twin ladder network is composed of two copies of the
    same network, each running separately on identical inputs.
    The final output is the difference between the two streams,
    which by construction should always be zero.

    We are going to make the two streams together.
    The first half of the variables correspond to the first stream
    The second half of the variables correspond to the second stream

    To account for numerical error, we will add a tiny bias to the output,
    and our final proof would be to show that we can't have a negative output.
    '''

    stream = []
    nb_inputs = ladder_dimension[0]
    nb_stream_out = ladder_dimension[-1]

    inp_domain = torch.Tensor([[-10, 10]]*nb_inputs)

    prev_size = nb_inputs
    for lay_out_size in ladder_dimension[1:]:
        stream.append(nn.Linear(prev_size, lay_out_size,
                                bias=True))
        prev_size = lay_out_size

    twin_net_layers = []
    # Add a linear layer duplicating the input
    dup_layer = nn.Linear(nb_inputs, 2* nb_inputs)
    dup_weight = dup_layer.weight
    dup_bias = dup_layer.bias
    dup_weight.zero_()
    dup_bias.zero_()
    dup_weight[:nb_inputs, :] = torch.eye(nb_inputs)
    dup_weight[-nb_inputs:, :] = torch.eye(nb_inputs)
    twin_net_layers.append(dup_layer)

    # Create linear layers that are block diagonals
    # with the same blocks
    prev_size = 2*nb_inputs
    for stream_lay in stream:
        nb_in = stream_lay.in_features
        nb_out = stream_lay.out_features
        twin_lay = nn.Linear(2*nb_in, 2*nb_out, bias=True)
        twin_lay_weight = twin_lay.weight
        twin_lay_bias = twin_lay.bias
        twin_lay_weight.zero_()

        twin_lay_weight[:nb_out, :nb_in].copy_(stream_lay.weight)
        twin_lay_weight[-nb_out:, -nb_in:].copy_(stream_lay.weight)
        twin_lay_bias[:nb_out].copy_(stream_lay.bias)
        twin_lay_bias[-nb_out:].copy_(stream_lay.bias)

        twin_net_layers.append(twin_lay)
        twin_net_layers.append(nn.ReLU())
        prev_size = 2*nb_out

    # Delete the last ReLU created
    del twin_net_layers[-1]
    # Create the final linear layers that merge the two streams back
    closing_layer = nn.Linear(prev_size, 1, bias=True)
    closing_layer.bias.fill_(margin)
    closing_layer.weight[0, :nb_stream_out].fill_(1)
    closing_layer.weight[0, -nb_stream_out:].fill_(-1)

    twin_net_layers.append(closing_layer)

    twin_ladder_net = nn.Sequential(*twin_net_layers)

    return twin_net_layers, inp_domain


def main():
    parser = argparse.ArgumentParser(description="Generate a twin-ladder network problem"
                                     "according to the dimension given as CLI arguments.")
    parser.add_argument('output_format', type=str,
                        help='Which format should the output we written in',
                        choices=["nnet", "rlv"])
    parser.add_argument('output_file', type=argparse.FileType('w'),
                        help="Where to write down the generated problem")
    parser.add_argument('margin', type=float,
                        help="What should the margin by which the property is true?")
    parser.add_argument('ladder_dims', type=int, nargs='+',
                        help="Dimension of each stage of the network")
    parser.add_argument('--seed', type=int,
                        default=0)

    args = parser.parse_args()
    assert len(args.ladder_dims) > 2, "Need several layers in the network"

    torch.manual_seed(args.seed)

    network_layers, domain = generate_network(args.ladder_dims, args.margin)

    network_layers = simplify_network(network_layers)
    assert_network_greater_than(nn.Sequential(*network_layers), domain, args.margin)

    if args.output_format == "nnet":
        dump_nnet(args.output_file, network_layers, domain)
    elif args.output_format == "rlv":
        dump_rlv(args.output_file, network_layers, domain)
    else:
        raise Exception("Unknown format")

if __name__ == '__main__':
    main()
