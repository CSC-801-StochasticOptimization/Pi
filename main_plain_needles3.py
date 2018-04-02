from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

__author__ = "bigfatnoob"

import lib
import argparse
import numpy as np


def get_file_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("-s", "--seedInit", type=int, default=None, help="Initial seed of experiment")
  parser.add_argument("-p", "--power", type=int, default=2, help="Max power of 10 to throw")
  parser.add_argument("-n", "--numThreads", type=int, default=8, help="Max number of threads")
  return parser.parse_args()


def main():
  args = get_file_args()
  powers = args.power
  seed_init = args.seedInit
  n_threads = args.numThreads
  if seed_init is None:
    seed_init = np.random.randint(0, 2 ** 16)
  np.random.seed(seed_init)
  print("# Seed Init = %d" % seed_init)
  print("# Max Throws = %d" % 10 ** powers)
  folder = "results-python/needles3/"
  lib.delete(folder)
  throws = [10 ** i for i in range(1, powers + 1)]
  for throw in throws:
    print("# Throw: %d" % throw)
    lib.throws_experiment(1.0, 1.0, lib.pi_throws_3d, throw, 100, folder=folder, save_file="triplegrid_%d.csv" % throw,
                          num_threads=n_threads)
  lib.aggregate_throws(throws, folder, '%sfg_asym_pi_plain_needles3_%d_%d.txt' % (folder, powers, seed_init))


def main_single():
  args = get_file_args()
  powers = args.power
  seed_init = args.seedInit
  n_threads = args.numThreads
  if seed_init is None:
    seed_init = np.random.randint(0, 2 ** 16)
  throw = 10 ** powers
  print("# Seed Init = %d" % seed_init)
  folder = "results-python/needles3/"
  print("# Throw: %d" % throw)
  lib.throws_experiment(1.0, 1.0, lib.pi_throws_3d, throw, 5, folder=folder, save_file="triplegrid_%d.csv" % throw,
                        num_threads=n_threads)
  throws = [10 ** i for i in range(1, powers + 1)]
  lib.aggregate_throws(throws, folder, '%sfg_asym_pi_plain_needles3_%d.txt' % (folder, powers))


if __name__ == "__main__":
  main_single()
