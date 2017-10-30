import bisect
import math
import torch


class CandidateDomain:
    '''
    Object representing a domain as produced by the BranchAndBound algorithm.
    Comparison between its elements is based on the values of the lower bounds
    that are estimated for it.
    '''
    def __init__(self, lb=-float('inf'), ub=float('inf'), dm=None):
        self.lower_bound = lb
        self.upper_bound = ub
        self.domain = dm

    def __lt__(self, other):
        return self.lower_bound < other.lower_bound

    def __le__(self, other):
        return self.lower_bound <= other.lower_bound

    def __eq__(self, other):
        return self.lower_bound == other.lower_bound

    def __repr__(self):
        string = f"[LB: {self.lower_bound:.4e}\t" \
                 f" UB:  {self.upper_bound:.4e}\n" \
                 f" Domain: {self.domain}]"
        return string

    def area(self):
        '''
        Compute the area of the domain
        '''
        dom_sides = self.domain.select(1, 1) - self.domain.select(1, 0)
        dom_area = dom_sides.prod()
        return dom_area


def bab(net, domain, eps=1e-3, decision_bound=None):
    '''
    Uses branch and bound algorithm to evaluate the global minimum
    of a given neural network.
    `net`           : Neural Network class, defining the `get_upper_bound` and
                      `get_lower_bound` functions
    `domain`        : Tensor defining the search bounds at each dimension.
    `eps`           : Maximum difference between the UB and LB over the minimum
                      before we consider having converged
    `decision_bound`: If not None, stop the search if the UB and LB are both
                      superior or both inferior to this value.

    Returns         : Lower bound and Upper bound on the global minimum,
                      as well as the point where the upper bound is achieved
    '''

    global_ub_point, global_ub = net.get_upper_bound(domain)
    global_lb = net.get_lower_bound(domain)

    nb_input_var = len(domain)
    normed_domain = torch.stack((torch.zeros(nb_input_var),
                                 torch.ones(nb_input_var)), 1)
    domain_lb = domain.select(1, 0)
    domain_width = domain.select(1, 1) - domain.select(1, 0)
    domain_lb = domain_lb.contiguous().view(nb_input_var, 1).expand(nb_input_var, 2)
    domain_width = domain_width.view(nb_input_var, 1).expand(nb_input_var, 2)

    # Use objects of type CandidateDomain to store domains with their bounds.
    candidate_domain = CandidateDomain(lb=global_lb, ub=global_ub,
                                       dm=normed_domain)
    domains = [candidate_domain]

    # This counter is used to decide when to prune domains
    prune_counter = 0

    while global_ub - global_lb > eps:
        # Pick a domain to branch over and remove that from our current list of
        # domains. Also potentially perform some pruning on the way.
        selected_candidate_domain = pick_out(domains, global_ub-eps)

        # Genearate new, smaller (normalized) domains using box split.
        ndoms = box_split(selected_candidate_domain.domain)

        for ndom_i in ndoms:
            # Find the upper and lower bounds on the minimum in dom_i
            dom_i = domain_lb + domain_width * ndom_i
            dom_ub_point, dom_ub = net.get_upper_bound(dom_i)
            dom_lb = net.get_lower_bound(dom_i)

            # Update the global upper if the new upper bound found is lower.
            if dom_ub < global_ub:
                global_ub = dom_ub
                global_ub_point = dom_ub_point

            # Add the domain to our current list of domains if its lowerbound
            # is less than the global upperbound.
            if dom_lb < global_ub:
                candidate_domain_to_add = CandidateDomain(lb=dom_lb,
                                                          ub=dom_ub,
                                                          dm=ndom_i)
                add_domain(candidate_domain_to_add, domains)
                prune_counter += 1

        # Prune domains whose lowerbounds are larger than or equal to the
        # global upperbound.
        # If domains list is larger than 100 items and if prune_counter has
        # reached a threshold, prune domains that we no longer need.
        if prune_counter >= 100 and len(domains) >= 100:
            # Remove domains with dom_lb >= global_ub
            domains = prune_domains(domains, global_ub-eps)
            prune_counter = 0

            # Do a pass over all the remaining domains to evaluate how much of
            # the input is there left to prune
            # print_remaining_domain(domains)
            print(f"Current: lb: {global_lb}\t ub: {global_ub}")

        # Update the global lower bound with a lower bound that belongs
        # to a domain in the updated list "domains" .
        # TODO: This current implementation is only a global lower bound
        #       if we sort domains by lower_bound.
        if len(domains) > 0:
            global_lb = domains[0].lower_bound
        else:
            # if there is no more domains, we have pruned them all.
            global_lb = global_ub - eps

        # Stopping criterion
        if (decision_bound is not None) and (global_lb >= decision_bound):
            break
        elif global_ub < decision_bound:
            break

    return global_lb, global_ub, global_ub_point


def add_domain(candidate, domains):
    '''
    Use binary search to add the new domain `candidate`
    to the candidate list `domains` so that `domains` remains a sorted list.
    '''
    bisect.insort_left(domains, candidate)


def pick_out(domains, threshold):
    '''
    Pick the first domain in the `domains` sequence
    that has a lower bound lower than `threshold`.

    Any domain appearing before the chosen one but having a lower_bound greater
    than the threshold is discarded.

    Returns: Non prunable CandidateDomain with the lowest reference_value.
    '''
    assert len(domains) > 0, "The given domains list is empty."
    while True:
        assert len(domains) > 0, "No domain left to pick from."
        selected_candidate_domain = domains.pop(0)
        if selected_candidate_domain.lower_bound < threshold:
            break

    return selected_candidate_domain


def box_split(domain):
    '''
    Use box-constraints to split the input domain.
    Split by dividing the domain into two from its longest edge.
    Assumes a rectangular domain, which is aligned with the cartesian
    coordinate frame.

    `domain`:  A 2d tensor whose rows contain lower and upper limits
               of the corresponding dimension.
    Returns: A list of sub-domains represented as 2d tensors.
    '''
    # Find the longest edge by checking the difference of lower and upper
    # limits in each dimension.
    diff = domain[:, 1] - domain[:, 0]
    edgelength, dim = torch.max(diff, 0)

    # Unwrap from tensor containers
    edgelength = edgelength[0]
    dim = dim[0]

    # Now split over dimension dim:
    half_length = edgelength/2

    # dom1: Upper bound in the 'dim'th dimension is now at halfway point.
    dom1 = domain.clone()
    dom1[dim, 1] -= half_length

    # dom2: Lower bound in 'dim'th dimension is now at haflway point.
    dom2 = domain.clone()
    dom2[dim, 0] += half_length

    sub_domains = [dom1, dom2]

    return sub_domains


def prune_domains(domains, threshold):
    '''
    Remove domain from `domains`
    that have a lower_bound greater than `threshold`
    '''
    # TODO: Could do this with binary search rather than iterating.
    # TODO: If this is not sorted according to lower bounds, this
    # implementation is incorrect because we can not reason about the lower
    # bounds of the domain that come after
    for i in range(len(domains)):
        if domains[i].lower_bound >= threshold:
            domains = domains[0:i]
            break
    return domains


def print_remaining_domain(domains):
    '''
    Iterate over all the domains, measuring the part of the whole input space
    that they contain and print the total share it represents.
    '''
    remaining_area = 0
    for dom in domains:
        remaining_area += dom.area()
    print(f'Remaining portion of the input space: {remaining_area*100:.8f}%')
