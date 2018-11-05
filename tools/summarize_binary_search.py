import argparse
import json
import os


def main():
  parser = argparse.ArgumentParser(description="Take a directory where a radius binary search result can be found"
                                   "and collect everything into a big table")
  parser.add_argument('radii_directory', type=str,
                      help='Directory containing all the results.')
  parser.add_argument('result_file', type=str,
                      help='Path where to store the agglomerate stuff')
  parser.add_argument('--keep_first', type=int,
                      help='Drop everything but the first X samples',
                      default=0)
  args = parser.parse_args()

  handled_samples = os.listdir(args.radii_directory)
  results = {}
  for sample in handled_samples:
    sample_results_path = os.path.join(args.radii_directory, sample, 'binary_search_result.txt')
    if not os.path.exists(sample_results_path):
      continue
    with open(sample_results_path, 'r') as sp_resfile:
      sp_res = sp_resfile.read()

    eps_lb, eps_ub = json.loads(sp_res.split('\n')[0])
    timing = float(sp_res.split('\n')[1])

    assert eps_ub - eps_lb < 1e-5

    results[int(sample)] = {
        "idx": int(sample),
        "timing": timing,
        "eps": (eps_ub + eps_lb) / 2
    }

  if args.keep_first != 0 and len(results) > args.keep_first:
    all_ids = sorted(results.keys())
    first_results = {}
    for idx in all_ids[:args.keep_first]:
      first_results[idx] = results[idx]
    results = first_results

  with open(args.result_file, 'w') as out_file:
    json.dump(results, out_file)

if __name__=='__main__':
  main()
