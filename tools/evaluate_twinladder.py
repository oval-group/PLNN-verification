#!/usr/bin/env python
import argparse
import math
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import re
from tools.compare_benchmarks import load_bench_results


class Settings:
    INP = "inp"
    LAYERS = "layers"
    WIDTH = "width"
    MARGIN = "margin"

    extract_re = re.compile('-(?P<nb_inp>.+)_inp-(?P<nb_lay>.+)_layers-(?P<width>.+)_width-(?P<margin>.+)_margin')

    def __init__(self, inp, layers, width, margin):
        self.inp = inp
        self.layers = layers
        self.width = width
        self.margin = margin

    @classmethod
    def from_name(cls, descr):
        match = re.search(cls.extract_re, descr)

        nb_inp = int(match["nb_inp"])
        nb_lay = int(match["nb_lay"])
        width = int(match["width"])
        margin = float(match["margin"])
        return cls(nb_inp, nb_lay, width, margin)

    def name_without_variation(self, variation=None):
        name = []
        if variation != self.INP:
            name.append(f"{self.inp}_inp")
        if variation != self.LAYERS:
            name.append(f"{self.layers}_layers")
        if variation != self.WIDTH:
            name.append(f"{self.width}_width")
        if variation != self.MARGIN:
            name.append(f"{self.margin}_margin")
        name = "|".join(name)
        return name

    def __repr__(self):
        return self.name_without_variation()

var2axis = {
    Settings.INP: "Number of Inputs",
    Settings.LAYERS: "Network Depth",
    Settings.WIDTH: "Width of hidden layers",
    Settings.MARGIN: "Property margin"
}
DIMS = [Settings.INP, Settings.LAYERS, Settings.WIDTH, Settings.MARGIN]


def parse_all_names(results_dir):
    new_res_dict = {}
    for name, result in results_dir.items():
        settings = Settings.from_name(name)
        new_res_dict[settings] = result
    return new_res_dict


def gather_fixed_along_variation(results_dir, variation):
    all_dps = {}
    for settings, res in results_dir.items():
        val_on_variation = getattr(settings, variation)
        curve_key = settings.name_without_variation(variation)

        if curve_key not in all_dps:
            all_dps[curve_key] = {}
        all_dps[curve_key][val_on_variation] = res[1]

    all_curves = {}
    for curve_key, curve_dps in all_dps.items():
        all_curves[curve_key] = tuple(zip(*sorted(curve_dps.items())))
    return all_curves


def filter_useless_curves(all_curves):
    useful_curves = {}
    for curve_key, (curve_x, curve_y) in all_curves.items():
        if all(map(lambda x: x == curve_y[0], curve_y)):
            pass
        else:
            useful_curves[curve_key] = (curve_x, curve_y)
    return useful_curves


def print_all_curves(all_curves):
    for curve_key, (curve_x, curve_y) in all_curves.items():
        print(curve_key)
        print(f"\t{curve_x}")
        print(f"\t{curve_y}")
        print("\n\n")


def plot_all_curves(all_curves, variation, target_filename):

    all_xy = list(all_curves.values())
    xs = all_xy[0][0]

    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot(1, 1, 1)

    if max(xs) / min(xs) > 10:
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim([0.1, 7300])

    for curve_key, (curve_x, curve_y) in all_curves.items():
        ax.plot(curve_x, curve_y, alpha=0.5, linewidth=2.0)
    ax.set_xlabel(var2axis[variation])
    ax.set_ylabel("Runtime (in s)")
    ax.grid(b=True, axis='both')

    target_format = "eps" if target_filename.endswith('.eps') else ".png"
    plt.savefig(target_filename, format=target_format, dpi=300)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the results of a "
                                     "given method along the various "
                                     "hyperparameters axis.")
    parser.add_argument('result_folder', type=str,
                        help="Results directory to analyze")
    parser.add_argument('variation_axis', type=str,
                        help="Along which dimension to observe the variation",
                        choices=DIMS)
    parser.add_argument('--remove_useless', action="store_true",
                        help="Don't show the curves that have no variation, "
                        "probably all stuck at timeout")
    parser.add_argument('--output_plot', type=str,
                        help="If a path to a file is passed, "
                        "save something there.")
    args = parser.parse_args()

    all_results = load_bench_results(args.result_folder)
    all_results = parse_all_names(all_results)

    all_curves = gather_fixed_along_variation(all_results, args.variation_axis)
    if args.remove_useless:
        all_curves = filter_useless_curves(all_curves)
    print_all_curves(all_curves)

    if args.output_plot is not None:
        plot_all_curves(all_curves, args.variation_axis, args.output_plot)

if __name__ == '__main__':
    main()
