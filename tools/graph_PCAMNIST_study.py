#!/usr/bin/env python
import argparse
import math
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import re
from tools.compare_benchmarks import load_bench_results

default_inp = 10
default_width = 25
default_depth = 4
default_margin = 1000

timeout_val = 7200

font = {'size': 28}

matplotlib.rc('font', **font)



class Settings:
    INP = "inp"
    WIDTH = "width"
    DEPTH = "depth"
    MARGIN = "margin"

    AXES_NAMES = ["Number of Inputs",
                  "Layer Width",
                  "Layer Depth",
                  'Satisfiability margin']

    extract_re = re.compile('--(?P<nb_inp>.+)_inp'
                            '--(?P<width>.+)_width'
                            '--(?P<depth>.+)_depth'
                            '-(?P<margin>.+)__margin')

    def __init__(self, inp, width, depth, margin):
        self.inp = inp
        self.width = width
        self.depth = depth
        self.margin = margin

    def params(self):
        return (self.inp, self.width, self.depth, self.margin)


    @classmethod
    def from_name(cls, descr):
        match = re.search(cls.extract_re, descr)

        nb_inp = int(match["nb_inp"])
        width = int(match["width"])
        depth = int(match["depth"])
        margin = float(match["margin"])
        return cls(nb_inp, width, depth, margin)

    def val(self, wanted_params):
        '''
        Returns None if it doesn't fit the wanted parameters.
        If it fits them, returns the value it has along the variation axis.
        '''
        var_val = None
        var_val_set = False
        for w_param, a_param in zip(wanted_params.params(),
                                    self.params()):
            if w_param != 0:
                if a_param != w_param:
                    return None
            else:
                assert not var_val_set
                var_val = a_param
                var_val_set = True
        if var_val_set:
            return var_val


def parse_all_names(results_dir, wanted_params):
    new_res_dict = {}
    for name, result in results_dir.items():
        settings = Settings.from_name(name)
        fit_val = settings.val(wanted_params)
        if fit_val is not None:
            # [1] because we don't want the statuses
            new_res_dict[fit_val] = result[1]
    return new_res_dict


def plot_all_curves(all_curves, var_axis_name, target_filename, log_x):
    all_methods = all_curves.keys()
    all_xs = []
    all_ys = []
    for method, scores in all_curves.items():
        all_xs.extend(scores.keys())

    fig = plt.figure(figsize=(10, 10))

    ax = plt.subplot(1, 1, 1)

    if log_x:
        if min(all_xs) < 0:
            # Need symmetric log because have negative stuff
            ax.set_xscale('symlog')
        else:
            ax.set_xscale('log')
    ax.set_yscale("log")
    ax.set_ylim([0.01, 10000])

    if var_axis_name == 'Satisfiability margin':
        ax.axhline(linewidth=5.0, y=2e-2, xmin=0, xmax=0.5, color="blue")
        ax.annotate('SAT / False', color="blue", xy=(0.2, 0.1), xycoords='axes fraction')
        ax.axhline(linewidth=5.0, y=2e-2, xmin=0.5, xmax=1, color="green")
        ax.annotate('UNSAT / True', color="green", xy=(0.6, 0.1), xycoords='axes fraction')

    for mtd in all_curves:
        xs = []
        ys = []
        for x in sorted(all_curves[mtd]):
            xs.append(x)
            ys.append(all_curves[mtd][x])

        ax.plot(xs, ys, linewidth=7.0, alpha=0.5, label=mtd)

    ax.axhline(linewidth=5.0, y=timeout_val, color='red', linestyle=':')
    ax.annotate('Timeout', xy=(0.9, 0.9), color='red',
                xycoords='axes fraction')

    ax.set_ylabel("Timing (in s.)")
    ax.set_xlabel(var_axis_name)
    ax.grid(b=True, axis='both')

    if var_axis_name == 'Number of Inputs':
        ax.legend(fontsize=30)
    target_format = "eps" if target_filename.endswith('.eps') else "png"
    plt.tight_layout()
    plt.savefig(target_filename, format=target_format, dpi=300)



def main():
    parser = argparse.ArgumentParser(description="Evaluate the results of several methods along"
                                     "the various hyperparameter axis of proofs")
    parser.add_argument('results_folder', type=str, nargs='+',
                        help="Results directory to compare")
    parser.add_argument('--inp', type=int,
                        default=default_inp,
                        help="Value of nb of inp to read for the graph. 0 for varying.")
    parser.add_argument('--width', type=int,
                        default=default_width,
                        help="Value of the layer width to read for the graph. 0 for varying.")
    parser.add_argument('--depth', type=int,
                        default=default_depth,
                        help="Value of the nb of layer to read for the graph. 0 for varying.")
    parser.add_argument('--margin', type=int,
                        default=default_margin,
                        help="Value of the margin to read for the graph. 0 for varying.")
    parser.add_argument('--log_x', action='store_true')
    parser.add_argument('--output_plot', type=str,
                        help="If a path to a file is passed, save something there.")
    args = parser.parse_args()

    # Check that there is only one axis of variation
    wanted_params = Settings(args.inp, args.width, args.depth, args.margin)
    if sum(elt==0 for elt in wanted_params.params()) != 1:
        raise Exception("Need one axis of variation")

    all_results = {}
    method_name_convert = {
        'grbBB': 'BlackBox',
        'AsymMIP': 'MIP-Planet'
    }
    for res_folder in args.results_folder:
        method_name = os.path.split(res_folder)[-1]
        if method_name in method_name_convert:
            method_name = method_name_convert[method_name]
        method_res = load_bench_results(res_folder)
        method_res = parse_all_names(method_res, wanted_params)
        all_results[method_name] = method_res

    variation_axis_pos = wanted_params.params().index(0)
    var_axis_name = Settings.AXES_NAMES[variation_axis_pos]

    if args.output_plot is not None:
        plot_all_curves(all_results, var_axis_name, args.output_plot, args.log_x)


if __name__ == "__main__":
    main()
