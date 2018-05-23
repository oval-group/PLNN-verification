import torch
import copy

from plnn.branch_and_bound import pick_out, add_domain, prune_domains
from torch import nn

class ReLUDomain:
    '''
    Object representing a domain where the domain is specified by decision
    assigned to ReLUs.
    Comparison between instances is based on the values of
    the lower bound estimated for the instances.

    The domain is specified by `mask` which corresponds to a pattern of ReLUs.
    Neurons mapping to a  0 value are assumed to always have negative input (0 output slope)
          "               1                    "             positive input (1 output slope).
          "               -1 value are considered free and have no assumptions.

    For a MaxPooling unit, -1 indicates that we haven't picked a dominating input
    Otherwise, this indicates which one is the dominant one
    '''
    def __init__(self, mask, lb=-float('inf'), ub=float('inf')):
        self.mask = mask
        self.lower_bound = lb
        self.upper_bound = ub

    def __lt__(self, other):
        return self.lower_bound < other.lower_bound

    def __le__(self, other):
        return self.lower_bound <= other.lower_bound

    def __eq__(self, other):
        return self.lower_bound == other.lower_bound


def relu_bab(net, domain, eps=1e-3, decision_bound=None):
    '''
    Uses branch and bound algorithm to evaluate the global minimum
    of a given neural network.
    `net`           : Neural Network class, defining the `get_upper_bound` and
                      `get_lower_bound` functions, supporting the `mask` argument
                      indicating the phase of the ReLU.
    `domain`        : Tensor defining the search bounds at each dimension.
    `eps`           : Maximum difference between the UB and LB over the minimum
                      before we consider having converged
    `decision_bound`: If not None, stop the search if the UB and LB are both
                      superior or both inferior to this value.

    Returns         : Lower bound and Upper bound on the global minimum,
                      as well as the point where the upper bound is achieved
    '''
    nb_visited_states = 0
    initial_mask = net.get_initial_mask()

    global_ub_point, global_ub = net.get_upper_bound(domain)
    global_lb, updated_mask = net.get_lower_bound(domain, initial_mask)

    candidate_domain = ReLUDomain(updated_mask, global_lb, global_ub)
    domains = [candidate_domain]

    prune_counter = 0

    while global_ub - global_lb > eps:
        # Pick a domain to branch over and remove that from our current list of
        # domains. Also, potentially perform some pruning on the way.
        candidate_domain = pick_out(domains, global_ub - eps)

        # Generate new, smaller domains by splitting over a ReLU
        n_masks = relu_split(net.layers, candidate_domain.mask)
        for n_mask_i in n_masks:
            # Find the upper and lower bounds on the minimum in the domain
            # defined by n_mask_i
            nb_visited_states += 1
            if (nb_visited_states % 10) == 0:
                print(f"Running Nb states visited: {nb_visited_states}")
            dom_lb, updated_mask = net.get_lower_bound(domain, n_mask_i)
            if len(n_masks) == 1:
                # The parent domain could not be splitted.
                # This means that all the non-linearities are fixed.
                # This means that computing the lower bound gives
                # exactly the minimum over this domain.
                dom_ub = dom_lb
                # The point can be obtained by looking at the gurobi model
                dom_ub_point = torch.Tensor([var.X for var in net.gurobi_vars[0]])
            else:
                dom_ub_point, dom_ub = net.get_upper_bound(domain)

            # Update the global upper if the new upper bound found is lower.
            if dom_ub < global_ub:
                global_ub = dom_ub
                global_ub_point = dom_ub_point

            # Add the domain to our current list of domains if its lowerbound
            # is less than the global upperbound.
            if dom_lb < global_ub:
                dom_to_add = ReLUDomain(updated_mask, lb=dom_lb, ub=dom_ub)
                add_domain(dom_to_add, domains)
                prune_counter += 1

        if prune_counter >= 100 and len(domains) >= 100:
            domains = prune_domains(domains, global_ub - eps)
            prune_counter = 0

            print(f"Current: lb:{global_lb}\t ub: {global_ub}")

        if len(domains) > 0:
            global_lb = domains[0].lower_bound
        else:
            # If there is no more domains, we have pruned them all
            global_lb = global_ub - eps

        # Stopping criterion
        if (decision_bound is not None) and (global_lb >= decision_bound):
            break
        elif global_ub < decision_bound:
            break

    return global_lb, global_ub, global_ub_point, nb_visited_states


def relu_split(layers, mask):
    '''
    Given a mask that defines a domain, split it according to a non-linerarity.

    The non-linearity is chosen to be as early as possible in the network, but
    this is just a heuristic.

    `layers`: list of layers in the network. Allows us to distinguish
              Maxpooling and ReLUs
    `mask`: A list of [list of {-1, 0, 1}] where each elements corresponds to a layer,
            giving constraints on the Neuron.
    Returns: A list of masks, in the same format

    '''
    done_split = False
    non_lin_layer_idx = 0
    all_new_masks = []
    for layer_idx, layer in enumerate(layers):
        if type(layer) in [nn.ReLU, nn.MaxPool1d]:
            non_lin_lay_mask = mask[non_lin_layer_idx]
            if done_split:
                # We have done our split, so no need for any additional split
                # -> Pass along all of the stuff
                for new_mask in all_new_masks:
                    new_mask.append(non_lin_lay_mask)
            elif all([neuron_dec != -1 for neuron_dec in non_lin_lay_mask]):
                # All the neuron in this layer have already an assumption.
                # This will just be passed along when we do our split.
                pass
            else:
                # This is the first layer we encounter that is not completely
                # assumed so we will take the first "undecided" neuron and
                # split on it.

                # Start by making two copies of everything that came before
                if type(layer) is nn.ReLU:
                    all_new_masks.append([])
                    all_new_masks.append([])
                elif type(layer) is nn.MaxPool1d:
                    for _ in range(layer.kernel_size):
                        all_new_masks.append([])
                else:
                    raise NotImplementedError

                for prev_lay_mask in mask[:non_lin_layer_idx]:
                    for new_mask in all_new_masks:
                        new_mask.append(prev_lay_mask)

                # Now, deal with the layer that we are actually splitting
                neuron_to_flip = non_lin_lay_mask.index(-1)
                for choice, new_mask in enumerate(all_new_masks):
                    # choice will be 0,1 for ReLU
                    # it will be 0, .. kernel_size-1 for MaxPool1d
                    mod_layer = non_lin_lay_mask[:]
                    mod_layer[neuron_to_flip] = choice
                    new_mask.append(mod_layer)

                done_split = True
            non_lin_layer_idx += 1
    for new_mask in all_new_masks:
        assert len(new_mask) == len(mask)
    if not done_split:
        all_new_masks = [mask]
    return all_new_masks
