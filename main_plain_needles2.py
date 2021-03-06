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
  return parser.parse_args()


def main():
  args = get_file_args()
  powers = args.power
  seed_init = args.seedInit
  if seed_init is None:
    seed_init = np.random.randint(0, 2 ** 16)
  np.random.seed(seed_init)
  print("# Seed Init = %d" % seed_init)
  print("# Max Throws = %d" % 10 ** powers)
  folder = "results-python/needles2/"
  lib.delete(folder)
  throws = [10 ** i for i in range(1, powers + 1)]
  for throw in throws:
    print("# Throw: %d" % throw)
    lib.throws_experiment(1.0, 1.0, lib.pi_throws_2d, throw, 100, folder=folder, save_file="triplegrid_%d.csv" % throw)
  lib.aggregate_throws(throws, folder, '%sfg_asym_pi_plain_needles2_%d_%d.txt' % (folder, powers, seed_init))


if __name__ == "__main__":
  main()
