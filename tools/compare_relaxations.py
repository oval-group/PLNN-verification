#!/usr/bin/env python
import argparse
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch

from plnn.branch_and_bound import box_split
from plnn.model import load_and_simplify
from plnn.network_linear_approximation import LinearizedNetwork
from torch.autograd import Variable
from timeit import default_timer as timer

font = {'size': 26}

matplotlib.rc('font', **font)

def make_plots(target_filename, measurements):
    (values, timings, names, dom_areas) = measurements
    target_format = "eps" if target_filename.endswith('.eps') else "png"

    markers = ["r-", "g-", "b-", "y-"]
    fig = plt.figure(figsize=(10, 10))

    ax_value = plt.subplot(1, 1, 1)
    ax_value.set_xscale("log")

    global_min = min(map(min, values))
    global_max = max(map(max, values))
    ratio = (global_max - global_min)  / abs(global_max)

    if ratio > 100:
        ax_value.set_yscale("symlog")

    for mtd_idx in range(len(names)):
        ax_value.plot(dom_areas, values[mtd_idx], markers[mtd_idx],
                      label=names[mtd_idx], linewidth=12.0)
    ax_value.set_ylabel("Lower bound")
    ax_value.set_xlabel("Relative area")
    ax_value.grid(b=True, axis='both')
    ax_value.legend()

    plt.savefig(target_filename, format=target_format, dpi=300)

    fig = plt.figure(figsize=(10, 5))
    ax_timings = plt.subplot(1, 1, 1)
    ax_timings.set_xscale("log")
    ax_timings.set_yscale("log")
    for mtd_idx in range(len(names)):
        ax_timings.plot(dom_areas, timings[mtd_idx], markers[mtd_idx],
                        label=names[mtd_idx])
    ax_timings.set_ylabel("Computation time (in s)")
    ax_timings.set_xlabel("Relative area")
    ax_timings.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5),
                      ncol=1, fontsize=36)
    ax_timings.grid(b=True, axis='both')
    ax_timings.legend()

    timings_filename = target_filename.replace("." + target_format, "timings." + target_format)
    plt.savefig(timings_filename, format=target_format, dpi=300)

def benchmark(rlv_infile, nb_splits, shrink_factor):
    '''
    Compare relaxations over the property/network defined by `rlv_infile`
    We heuristically determine a "minimum" by sampling, and get a rectangular
    domain around it.
    For `nb_splits` time, we shrink all edges by `shrink_factor`, making the domain
    tighter and therefore the lower bound better.

    Returns:
    values:  List of [list of lowerbounds], one for each method compared
    timings: List of [list of timings], one for each method compared
    names:   List of Names of method
    areas:   List of the areas of the domains on which we computed the LBs.
    '''
    # Loading / Parsing of the networks
    rlv_infile.seek(0)
    lin_network, domain = load_and_simplify(rlv_infile, LinearizedNetwork)

    rlv_infile.seek(0)
    lin_network_noredefine, _ = load_and_simplify(rlv_infile, LinearizedNetwork)
    lin_network_noredefine.get_lower_bound = lin_network_noredefine.compute_lower_bound

    to_compare = [lin_network, lin_network_noredefine]
    names = ["ReApproximating", "FixedApproximation"]

    nb_methods = len(names)

    # Definition of the problem
    nb_input_var = domain.size(0)
    normed_domain = torch.stack((torch.zeros(nb_input_var),
                                 torch.ones(nb_input_var)), 1)
    domain_lb = torch.Tensor([lb for lb, _ in domain])
    domain_ub = torch.Tensor([ub for _, ub in domain])

    # Create the arrays that are going to hold what we are going to measure
    values = [[] for _ in range(nb_methods)]
    timings = [[] for _ in range(nb_methods)]

    dom_areas = []

    ndom = normed_domain
    dom = torch.stack([domain_lb, domain_ub], dim=1)

    ## Compute an estimate of where the minimum is
    best_min = float('inf')
    min_point = torch.zeros(nb_input_var)
    last_improvement = 0
    while(last_improvement <= 100):
        point_cd, min_cd = lin_network.get_upper_bound(dom)
        if min_cd <= best_min:
            min_point = point_cd
            best_min = min_cd
            last_improvement = 0
        last_improvement += 1

    print(f"Estimated minimum is {best_min}")

    # We are going to center the sequential regions around this point so that
    # they will focus always on the same point
    initial_width = torch.min(domain_ub - min_point,
                              min_point - domain_lb)
    center = min_point.view(nb_input_var, 1).expand(nb_input_var, 2)
    bounds = torch.stack([-initial_width, initial_width], dim=1)
    initial_domain = center + bounds
    lin_network_noredefine.define_linear_approximation(initial_domain)
    area = 1
    for i in range(nb_splits):
        dom = center + bounds
        for mtd_idx in range(nb_methods):
            net = to_compare[mtd_idx]

            start = timer()
            lb = net.get_lower_bound(dom)
            end = timer()
            print(f"{names[mtd_idx]} \t {lb} \t {end - start}")

            values[mtd_idx].append(lb)
            timings[mtd_idx].append(end-start)

        ## Store everything into the lists
        dom_areas.append(area)

        ## Shrink on the feasible domain
        for _ in range(nb_input_var):
            area *= shrink_factor
        bounds = bounds * shrink_factor

    all_measurements = (values, timings, names, dom_areas)
    return all_measurements


def main():
    parser = argparse.ArgumentParser(description="Read a .rlv file"
                                     "and compare various relaxations on it")
    parser.add_argument('rlv_infile', type=argparse.FileType('r'),
                        help='.rlv file to test relaxation on.')
    parser.add_argument('nb_splits', type=int,
                        help="How many time should we split?")
    parser.add_argument('shrink_factor', type=float,
                        help="By how much do we shrink the domain.")
    parser.add_argument('plot_outfile', type=str)
    args = parser.parse_args()

    measurements = benchmark(args.rlv_infile, args.nb_splits, args.shrink_factor)
    make_plots(args.plot_outfile, measurements)

if __name__ == '__main__':
    main()
