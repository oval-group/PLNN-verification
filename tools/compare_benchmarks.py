#!/usr/bin/env python
import argparse
import glob
import itertools
import os
import re
from collections import Counter

FAILS = ['timeout', 'OOM', 'ERROR']
SAT = 'SAT'
UNSAT = 'UNSAT'
ERROR = 'ERROR'
timeout_time = 7200

reluplex_log_re = re.compile('Total visited states:\s*(?P<nb_visited_states>\d+)')
standard_log_re = re.compile('Nb states visited:\s*(?P<nb_visited_states>\d+)(\.0)?$')
samples_log_re = re.compile('Nb samples:\s*(?P<nb_visited_states>\d+)')

def load_bench_results(bench_folder):
    all_results = {}

    all_dumps = glob.glob(f"{bench_folder}/**/*.txt", recursive=True)
    if len(all_dumps) == 0:
        print("No results found, maybe check the path to the results directory?")
    for dp in all_dumps:
        with open(dp, 'r') as dp_file:
            dp_content = dp_file.read()
        dp_id = dp.replace(bench_folder, '')
        dp_status, dp_time = dp_content.split()

        if dp_status == 'Done':
            # The optimization was finished, get more information
            # Get a more detailed status.
            if os.path.exists(dp + '.final'):
                path_to_decision = dp + '.final'
            else:
                path_to_decision = dp + '.log'

            with open(path_to_decision, 'r') as dec_file:
                dec_content = dec_file.read()

            if 'UNSAT' in dec_content:
                dp_status = UNSAT
            elif 'SAT' in dec_content:
                dp_status = SAT
            elif 'Error' in dec_content:
                dp_status = ERROR
                # dp_time = timeout_time
            else:
                raise Exception(f"Unknown status in file {path_to_decision}:\n{dec_content}")
        else:
            dp_time = timeout_time

        ## Collect the info on the number of splits
        nb_states = None
        is_reluplex = 'reluplex' in bench_folder
        is_noformal = 'noFormal' in bench_folder
        if is_reluplex:
            if dp_status != 'timeout':
                # There should be a '.final' file, where the last element of the
                # line is the number of splits
                path_to_final = dp + '.final'
                with open(path_to_final, 'r') as final:
                    dec_content = final.read()
                nb_states = int(dec_content.split(',')[-1])
            else:
                # There is no '.final' file but there should be a log file.
                # We can go look at the last printout of statistics and query this
                path_to_log = dp + '.log'
                with open(path_to_log, 'r') as logfile:
                    for line in logfile.readlines():
                        if line.strip().startswith('So far'):
                            match = re.search(reluplex_log_re, line)
                            nb_states = int(match["nb_visited_states"])
        else:
            path_to_out = dp + ".log"
            with open(path_to_out, 'r') as outfile:
                for line in outfile.readlines():
                    if is_noformal:
                        match = re.search(samples_log_re, line.strip())
                    else:
                        match = re.search(standard_log_re, line.strip())
                    if match is not None:
                        nb_states = int(match["nb_visited_states"])
                        if nb_states == 0:
                            # We're considering the number of visited nodes,
                            # so there is at least one
                            nb_states = 1

        all_results[dp_id] = (dp_status, float(dp_time), nb_states)

    return all_results


def remove_fails(res_dict):
    return {p_id: res
            for (p_id, res) in res_dict.items()
            if res[0] not in FAILS}


def remove_non_common(all_results):
    all_keys = [set(res.keys()) for res in all_results.values()]

    common_keys = all_keys[0]
    for keyset in all_keys[1:]:
        common_keys = common_keys.intersection(keyset)

    common_results = {method: {k: res[k] for k in common_keys}
                      for method, res in all_results.items()}
    print(f"Total number of properties after removing non-common: {len(common_keys)}")
    return common_results


def count_fails(d, gt_map):
    fail_count = 0
    for k, res in d.items():
        if res[0] in FAILS:
            fail_count += 1
        elif res[0] != gt_map[k]:
            fail_count += 1
    return fail_count

def count_error(d, gt_map):
    error_count = 0
    for k, res in d.items():
        if res[0] == ERROR:
            error_count += 1
        elif res[0] not in FAILS and res[0] != gt_map[k]:
            error_count += 1
    return error_count




def win_count(all_results, gt_map):
    all_cases = gt_map.keys()
    win_counts = Counter()

    for k in all_cases:
        all_scores = {method: all_results[method][k] for method in all_results
                      if k in all_results[method]}
        # Remove Fails and Wrong
        all_done_scores = {method: score[1] for method, score in all_scores.items()
                           if score[0] == gt_map[k]}
        if len(all_done_scores) == 0:
            continue
        else:
            best_method = min(all_done_scores, key=all_done_scores.get)
            best_score = all_done_scores[best_method]
            all_done_scores[best_method] = float('inf')
            second_best_method = min(all_done_scores, key=all_done_scores.get)
            second_best_score = all_done_scores[second_best_method]

            if len(all_scores) == len(all_done_scores):
                # All methods solved it
                if (second_best_score - best_score) / best_score < 1e-2:
                    best_method = "Ties"
            win_counts[best_method] += 1
            # print(f"Case: {k} -> {best_method}")
    return win_counts


def uniquely_solved(d1, d2):
    '''
    Print the list of properties solved by d1 and not by d2
    '''
    shared_keys = set(d1.keys()).intersection(set(d2.keys()))
    uniq_solved = []
    for k in shared_keys:
        if (d1[k][0] not in FAILS) and (d2[k][0] in FAILS):
            uniq_solved.append(k)
    uniq_solved.sort()
    for prop in uniq_solved:
        print(prop)


def average_runtime(d):
    return sum(res[1] for res in d.values()) / len(d)


def build_GT_table(all_results, all_unsat=False):
    per_method_dec = {}

    for method, results in all_results.items():
        for case, (status, timing, nb_nodes) in results.items():
            if case not in per_method_dec:
                per_method_dec[case] = {}
            per_method_dec[case][method] = status
            if all_unsat and (status not in FAILS) and (status != UNSAT):
                print(f"Method {method} is wrong on {case}")

    gt_table = {}
    for case, m2dec_map in per_method_dec.items():
        if all_unsat:
            gt_table[case] = UNSAT
        elif "__margin" in case:
            if "depth--" in case:
                # There is one separating dash, the other is the minus sign
                gt_table[case] = SAT
            else:
                # There is only the separating dash, the margin is positive
                # That means the property is True, problem is unsatisfiable
                gt_table[case] = UNSAT
        else:
            for method, dec in m2dec_map.items():
                if dec == ERROR:
                    print(f"**WARNING** Method {method} is wrong on {case}")
            all_decs = [dec for dec in m2dec_map.values() if dec not in FAILS]
            if len(all_decs) == 0:
                # No method managed to solve this problem
                gt_table[case] = None
            else:
                vote_count = Counter(all_decs)
                if len(vote_count) > 1:
                    print(f"**WARNING** Not all methods agree on {case}")
                gt_table[case] = vote_count.most_common(1)[0][0]
    return gt_table


def main():
    parser = argparse.ArgumentParser(description="Compare the results"
                                     "on a set of benchmarks.")
    parser.add_argument('results_folder', type=str, nargs='+',
                        help="Results directory to compare.")
    parser.add_argument('--all_unsat', action='store_true',
                        help="If you know that all outputs are UNSAT, warn for incorrect assumptions")
    parser.add_argument('--only_success', action='store_true',
                        help="Ignore the properties that were not succesfully"
                        "solved.")
    parser.add_argument('--only_common', action='store_true',
                        help="Only do the comparison on the properties that"
                        "all solvers have attempted")

    args = parser.parse_args()

    common_prefix = os.path.commonprefix(args.results_folder)
    make_name = lambda x: x.replace(common_prefix, '')[:-1]

    all_results = {make_name(res_folder): load_bench_results(res_folder)
                   for res_folder in args.results_folder}

    if args.only_success:
        for method_name, results in all_results.items():
            all_results[method_name] = remove_fails(results)

    if args.only_common:
        all_results = remove_non_common(all_results)

    gt_table = build_GT_table(all_results, args.all_unsat)

    gt_table_only_SAT = {case: dec for case, dec in gt_table.items()
                         if dec == SAT}
    gt_table_only_UNSAT = {case: dec for case, dec in gt_table.items()
                           if dec == UNSAT}
    gt_table_all_none = {case: dec for case, dec in gt_table.items()
                         if dec is None}
    # Compute the ratio of Fails for each
    print("=== Proportion of successfully solved test cases")
    for method, results in all_results.items():
        error_count = count_error(results, gt_table)
        fail_count = count_fails(results, gt_table)
        nb_prop = len(results)
        print(f"{method}:\n\tSolved: {nb_prop - fail_count}"
              f"\tError: {error_count}"
              f"\tAttempted: {nb_prop}"
              f"\tSuccess rate: {100*(1-fail_count/nb_prop):.2f}"
              f"\tError rate: {100*(error_count / nb_prop):.2f}")
    print(f"{len(gt_table_all_none)} properties solved by no methods.")

    # Compute the numbers of wins
    print("\n\n")
    print("=== Number of wins")
    win_per_methods = win_count(all_results, gt_table)
    for method, mtd_win_count in sorted(win_per_methods.items()):
        print(f"{method} is the fastest in {mtd_win_count} cases")
    if not args.all_unsat:
        print("\n=== Number of wins on SAT problems")
        sat_win_per_methods = win_count(all_results, gt_table_only_SAT)
        for method, mtd_win_count in sorted(sat_win_per_methods.items()):
            print(f"{method} is the fastest in {mtd_win_count} cases")
    print("\n=== Number of wins on UNSAT problems")
    unsat_win_per_methods = win_count(all_results, gt_table_only_UNSAT)
    for method, mtd_win_count in sorted(unsat_win_per_methods.items()):
        print(f"{method} is the fastest in {mtd_win_count} cases")


    # # Compute the numbers of uniquely solved
    # print("\n\n")
    # print
    # print(f"=== Uniquely solved by {first_name}")
    # uniquely_solved(first, second)
    # print(f"=== Uniquely solved by {second_name}")
    # uniquely_solved(second, first)

    # Compute the average timings
    print("\n\n")
    if args.only_success and (not args.only_common):
        print("(Warning: This is an average runtimes only on successfully"
              "solved instances, which might be misleading)")
    if (not args.only_success):
        print("(Warning: Property having timed out were reported as their timeout)")
    if args.only_common:
        print("(Warning: This reports is not on all samples, only those"
              "that were in common between the two methods)")
    print("=== Average runtimes")
    for method, results in sorted(all_results.items()):
        mtd_avg_timing = average_runtime(results)
        print(f"{method} average runtime: {mtd_avg_timing} s")
    if not args.all_unsat:
        print("\n=== Average runtimes on SAT problems")
        for method, results in sorted(all_results.items()):
            sat_results = {k: results[k] for k in gt_table_only_SAT
                           if k in results}
            if len(sat_results) > 0:
                mtd_avg_timing = average_runtime(sat_results)
                print(f"{method} average runtime: {mtd_avg_timing} s")
    print("\n=== Average runtimes on UNSAT problems")
    for method, results in sorted(all_results.items()):
        unsat_results = {k: results[k] for k in gt_table_only_UNSAT
                         if k in results}
        if len(unsat_results) > 0:
            mtd_avg_timing = average_runtime(unsat_results)
            print(f"{method} average runtime: {mtd_avg_timing} s")

if __name__ == '__main__':
    main()
