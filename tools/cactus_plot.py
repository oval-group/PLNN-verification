#!/usr/bin/env python
import argparse
import os

import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt

from compare_benchmarks import load_bench_results, build_GT_table, remove_non_common

FAILS = ['timeout', 'OOM', 'ERROR']
SAT = 'SAT'
UNSAT = 'UNSAT'
ERROR = 'ERROR'
timeout_time = 7200

font = {'size': 25}
matplotlib.rc('font', **font)

def make_cactus_plot(all_results, gt_table, target_filename,
                     plot_nb_splits=False, log_scale_plot=False,
                     only=None, legend=None):

    nb_methods = len(all_results)

    ### Setting up the data
    adjusted_timings = {}

    for method, m_results in all_results.items():
        adjusted_timings[method] = []
        for prop, res in m_results.items():
            if res[0] in FAILS or (res[0] != gt_table[prop]):
                print(f"[{method}] Error {res[0]} on property {prop}")
                timing = float('inf')
            else:
                if plot_nb_splits:
                    timing = res[2]
                else:
                    timing = res[1]

            if only is None:
                adjusted_timings[method].append(timing)
            else:
                if gt_table[prop] == only:
                    adjusted_timings[method].append(timing)

        adjusted_timings[method].sort()

    nb_all = len(gt_table)

    ### Plot generation
    target_format = "eps" if target_filename.endswith('.eps') else "png"

    colors = matplotlib.colors.TABLEAU_COLORS
    method_2_color = {}
    for method, color in zip(all_results.keys(), colors.keys()):
        method_2_color[method] = color

    fig = plt.figure(figsize=(10,10))
    ax_value = plt.subplot(1, 1, 1)
    ax_value.axhline(linewidth=3.0, y=100, linestyle='dashed', color='grey')

    y_min = 0
    y_max = 100
    ax_value.set_ylim([y_min, y_max+5])

    min_solve = float('inf')
    max_solve = float('-inf')
    for timings in adjusted_timings.values():
        min_solve = min(min_solve, min(timings))
        finite_vals = [val for val in timings if val != float('inf')]
        if len(finite_vals) > 0:
            max_solve = max(max_solve, max([val for val in timings if val != float('inf')]))

    if log_scale_plot:
        ax_value.set_xscale("log")
        axis_min = min(0.5 * min_solve, 1)
    else:
        axis_min = 0

    if plot_nb_splits:
        axis_max = max_solve
    else:
        axis_max = timeout_time
    ax_value.set_xlim([axis_min, axis_max])

    # Plot all the properties
    linestyle_all = 'solid'
    for method, m_timings in adjusted_timings.items():
        # Make it an actual cactus plot
        xs = [axis_min]
        ys = [y_min]
        prev_y = 0
        for i, x in enumerate(m_timings):
            if x <= axis_max:
                # Add the point just before the rise
                xs.append(x)
                ys.append(prev_y)
                # Add the new point after the rise, if it's in the plot
                xs.append(x)
                new_y = 100*(i+1)/len(m_timings)
                ys.append(new_y)
                prev_y = new_y
        # Add a point at the end to make the graph go the end
        xs.append(axis_max)
        ys.append(prev_y)

        ax_value.plot(xs, ys, color=method_2_color[method],
                      linestyle=linestyle_all, label=method, linewidth=5.0)

    ax_value.set_ylabel("% of properties verified")
    if plot_nb_splits:
        ax_value.set_xlabel("Number of Nodes visited")
    else:
        ax_value.set_xlabel("Computation time (in s)")
    if legend is not None:
        ax_value.legend(loc=legend, fontsize=25)

    plt.savefig(target_filename, format=target_format, dpi=300)


def main():
    parser = argparse.ArgumentParser(description="Compare the results on a set of benchmarks.")
    parser.add_argument('results_folder', type=str, nargs='+',
                        help="Results directory to compare.")
    parser.add_argument('--truth_folder', type=str,
                        help="Directory to use as a source of ground truth for SAT/UNSAT")
    parser.add_argument('--out_file' , type=str,
                        help="Where to generate the image")
    parser.add_argument('--plot_nb_splits', action="store_true",
                        help="Plot the number of splits done rather than the time.")
    parser.add_argument('--log_scale_plot', action="store_true",
                        help="Use a logscale on the x-axis")
    parser.add_argument('--only_common', action ="store_true",
                        help="Keep only the properties that all methods have run on")
    parser.add_argument('--only', type=str,
                        help="Keep only the properties that the GT classifies as this")
    parser.add_argument('--legend', type=str,
                        help="Where to put the legend")

    args = parser.parse_args()

    common_prefix = os.path.commonprefix(args.results_folder)
    make_name = lambda x: x.replace(common_prefix, '')[:-1]

    all_results = {make_name(res_folder): load_bench_results(res_folder)
                   for res_folder in args.results_folder}

    if args.only_common:
        all_results = remove_non_common(all_results)

    if args.truth_folder is not None:
        gt_results = load_bench_results(args.truth_folder)
        gt_table = {}
        for key, val in gt_results.items():
            gt_table[key] = val[0]
    else:
        gt_table = build_GT_table(all_results, False)

    make_cactus_plot(all_results, gt_table, args.out_file,
                     plot_nb_splits=args.plot_nb_splits,
                     log_scale_plot=args.log_scale_plot,
                     only=args.only, legend=args.legend)

    if args.plot_nb_splits:
        # If we're interested in the splits,
        # we might as well get the average time per splits
        nb_splits_per_method = {method: 0 for method in all_results.keys()}
        time_per_method = {method: 0 for method in all_results.keys()}
        for method, m_results in all_results.items():
            for (status, time, nb_nodes) in m_results.values():
                nb_splits_per_method[method] += nb_nodes
                time_per_method[method] += time
        txt_outfile_path = ".".join(args.out_file.split('.')[:-1] + ['txt'])
        with open(txt_outfile_path, 'w') as txt_outfile:
            txt_outfile.write('Method\tTotal time\tTotal nodes\tAvg Time per Node\n')
            for method in all_results.keys():
                total_time = time_per_method[method]
                total_nodes = nb_splits_per_method[method]

                node_per_s = total_time / total_nodes

                txt_outfile.write(f'{method}\t{total_time}\t{total_nodes}\t{node_per_s}\n')

if __name__=='__main__':
    main()
