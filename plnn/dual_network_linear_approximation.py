import torch
from torch import nn
from convex_adversarial import DualNetwork
from plnn.network_linear_approximation import LinearizedNetwork
from plnn.model import reluify_maxpool, simplify_network

class LooseDualNetworkApproximation(LinearizedNetwork):
    def __init__(self, layers):
        '''
        layers: A list of Pytorch layers containing only Linear/ReLU
        '''
        self.layers = layers

    def remove_maxpools(self, domain):
        if any(map(lambda x: type(x) is nn.MaxPool1d, self.layers)):
            new_layers = simplify_network(reluify_maxpool(self.layers, domain))
            self.layers = new_layers

    def get_lower_bounds(self, domains):
        '''
        Update the linear approximation for `domains` of the network and use it
        to compute a lower bound on the minimum of the output.

        domain: Tensor containing in each row the lower and upper bound for
                the corresponding dimension
                Batch_idx x dimension x bound_type (0 -> lb, 1 -> ub)
        '''
        # Okay, this is a disgusting, disgusting hack. This DEFINITELY should
        # be replaced by something more proper in practice but I'm doing this
        # for a quick experiment.

        # The code from https://github.com/locuslab/convex_adversarial only
        # works in the case of adversarial examples, that is, it assumes the
        # domain is centered around a point, and is limited by an infinity norm
        # constraint. Rather than properly implementing the general
        # optimization, I'm just going to convert the problem into this form,
        # by adding a fake linear at the beginning. This is definitely not
        # clean :)

        # widths =
        batch_size = domains.size(0)
        domain_lb = domains.select(2, 0)
        domain_ub = domains.select(2, 1)

        with torch.no_grad():
            x = (domain_ub + domain_lb) / 2
            domain_radius = (domain_ub - domain_lb)/2

            # assert (domain_radius[0] - domain_radius[1]).abs().sum() < 1e-6
            domain_radius = domain_radius[0]

            # Disgusting hack number 2:
            # In certain case we don't want to allow a variable to move.
            # Let's just allow it to move a tiny tiny bit
            domain_radius[domain_radius == 0] = 1e-6

            bias = x[0].clone()
            x[0].fill_(0)
            x[1] = (x[1] - bias) / domain_radius

            inp_layer = nn.Linear(domains.size(1), domains.size(1), bias=True)
            inp_layer.weight.copy_(torch.diag(domain_radius))
            inp_layer.bias.copy_(bias)
            fake_net = nn.Sequential(*simplify_network([inp_layer]
                                                       + self.layers))

            dual = DualNetwork(fake_net, x, 1)

            # since we have already encoded the property as a network layer,
            # our c vector in the kw sense is just a constant.
            bound_val = dual(torch.ones(batch_size, 1, 1))

            bound_val = bound_val.squeeze_(1)
            del dual

        return bound_val
