import torch
from torch import nn
from convex_adversarial import DualNetwork
from convex_adversarial.dual_layers import DualLinear, DualReLU
from plnn.network_linear_approximation import LinearizedNetwork
from plnn.model import reluify_maxpool, simplify_network


class LooseDualNetworkApproximation(LinearizedNetwork):
    def __init__(self, layers):
        '''
        layers: A list of Pytorch layers containing only Linear/ReLU
        '''
        self.layers = layers

    def remove_maxpools(self, domain, no_opt=False):
        if any(map(lambda x: type(x) is nn.MaxPool1d, self.layers)):
            new_layers = simplify_network(reluify_maxpool(self.layers, domain, no_opt))
            self.layers = new_layers

    def get_lower_bounds(self, domains):
        '''
        Create the linear approximation for `domains` of the network and use it
        to compute a lower bound on the minimum of the output.

        domain: Tensor containing in each row the lower and upper bound for
                the corresponding dimension
                Batch_idx x dimension x bound_type (0 -> lb, 1 -> ub)
        '''
        batch_size = domains.size(0)
        dual = self.build_approximation(domains)

        # since we have already encoded the property as a network layer,
        # our c vector in the kw sense is just a constant.
        bound_val = dual(torch.ones(batch_size, 1, 1))

        bound_val = bound_val.squeeze_(1)
        del dual

        return bound_val

    def get_intermediate_bounds(self, domain):
        '''
        Create the linear approximation and return all the intermediate bounds.
        '''
        batch_domain = domain.unsqueeze(0)
        dual = self.build_approximation(batch_domain)

        # Let's get the intermediate bounds, in the same way that they are
        # obtained by the other methods.
        lower_bounds = []
        upper_bounds = []
        # Add the input bounds
        lower_bounds.append(domain[:, 0])
        upper_bounds.append(domain[:, 1])

        for layer in dual.dual_net[1:]:
            if isinstance(layer, DualLinear):
                # Skip this one, we're going to get his bound
                # with the next DualReLU
                pass
            elif isinstance(layer, DualReLU):
                # K&W has this as input bounds to the ReLU, but our
                # codes reasons in terms of output bounds of a layer
                # We get this bounds and enqueue them, they correspond
                # to the output of the ReLU from before.
                lower_bounds.append(layer.zl.squeeze())
                upper_bounds.append(layer.zu.squeeze())

                # Let's also trivially determine what are the bounds on the
                # outputs of the ReLU
                lower_bounds.append(torch.clamp(layer.zl, 0).squeeze())
                upper_bounds.append(torch.clamp(layer.zu, 0).squeeze())
            else:
                raise NotImplementedError("Unplanned layer type.")

        # Also add the bounds on the final thing
        lower_bounds.append(dual(torch.ones(1,1,1)).squeeze())
        upper_bounds.append(-dual(-torch.ones(1,1,1)).squeeze())
        return lower_bounds, upper_bounds

    def build_approximation(self, domains):
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
        batched = domains.shape[0] > 1

        domain_lb = domains.select(2, 0)
        domain_ub = domains.select(2, 1)

        with torch.no_grad():
            x = (domain_ub + domain_lb) / 2
            domain_radius = (domain_ub - domain_lb)/2

            if batched:
                # Verify that we can use the same epsilon for both parts
                assert (domain_radius[0] - domain_radius[1]).abs().sum() < 1e-6
                # We have written the code assuming that the batch size would
                # be limited to 2, check that it is the case.
                assert domains.shape[0] <= 2

            domain_radius = domain_radius[0]

            # Disgusting hack number 2:
            # In certain case we don't want to allow a variable to move.
            # Let's just allow it to move a tiny tiny bit
            domain_radius[domain_radius == 0] = 1e-6

            bias = x[0].clone()
            x[0].fill_(0)
            if batched:
                x[1] = (x[1] - bias) / domain_radius

            inp_layer = nn.Linear(domains.size(1), domains.size(1), bias=True)
            inp_layer.weight.copy_(torch.diag(domain_radius))
            inp_layer.bias.copy_(bias)
            fake_net = nn.Sequential(*simplify_network([inp_layer]
                                                       + self.layers))

            dual = DualNetwork(fake_net, x, 1)

        return dual

