from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

__author__ = "bigfatnoob"

import lib
import argparse
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd

CNT_PROBE_LIMITS = {
    1: 178.2,
    2: 826.2,
    3: 3830.1,
    4: 17756.5,
    5: 82319.1,
    6: 381631.3,
    7: 1769243.0,
    8: 8202208.0
}


def get_file_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("-s", "--seedInit", type=int, default=None, help="Initial seed of experiment")
  parser.add_argument("-d", "--digits", type=int, default=3, help="Max significant digits")
  parser.add_argument("-p", "--samples", type=int, default=100, help="Number of samples in the experiment")
  return parser.parse_args()


def main():
  args = get_file_args()
  max_signif_digits = args.digits
  samples = args.samples
  seed_init = args.seedInit
  num_cores = multiprocessing.cpu_count()
  if seed_init is None:
    seed_init = np.random.randint(0, 2 ** 16)
  np.random.seed(seed_init)
  print("# Seed Init = %d" % seed_init)
  print("# Max Signif Digit = %d" % max_signif_digits)
  print("# Samples = %d" % samples)
  print("# Cores = %d" % num_cores)
  print("# Solver = Needles3")
  results = pd.DataFrame(columns=["seedInit", "solverName", "signifDigits", "piHat", "OFtol", "error", "isCensored",
                                  "numThrows", "runtime"])
  counter = 1
  for signif_digit in range(2, max_signif_digits + 1):
    print("### For %d significant digits" % signif_digit)
    seeds = np.random.randint(0, 2 ** 16, samples)
    signif_results = Parallel(n_jobs=num_cores)(delayed(
      lib.pi_needle_triple)(1, 1, cnt_probe_limit=CNT_PROBE_LIMITS[signif_digit],
                                  signif_digits=signif_digit, seed=seed, reject_censored=True) for seed in seeds)
    for i in range(len(signif_results)):
      results.loc[i + counter] = signif_results[i]
    counter += len(signif_results)
  results.index.name = 'sampleId'
  results.columns.name = results.index.name
  txt_file_name = 'results-python/fg_asym_pi_signif_needles3_%d_%d.txt' % (max_signif_digits, seed_init)
  html_file_name = 'results-python/fg_asym_pi_signif_needles3_%d_%d.html' % (max_signif_digits, seed_init)
  results.to_csv(txt_file_name, sep="\t")
  results.to_html(html_file_name)
  with open(html_file_name) as f:
    html_data = f.read()
    html_data += "\n"
  with open(html_file_name, "wb") as f:
    f.write(html_data)

if __name__ == "__main__":
  main()
