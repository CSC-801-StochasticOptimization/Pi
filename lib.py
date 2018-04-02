from __future__ import print_function, division
import sys
import os

sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True


__author__ = "bigfatnoob"

from matplotlib import pyplot as plt
import numpy as np
from shapely.geometry import Polygon, LineString, Point
import time
import pandas as pd
import csv
from joblib import Parallel, delayed
import multiprocessing
import shutil
import collections


def file_exists(file_name):
  """
  Check if file or folder exists
  :param file_name: Path of the file
  :return: True/False
  """
  return os.path.exists(file_name)


def mkdir(directory):
  """
  Create Directory if it does not exist
  """
  if not file_exists(directory):
    try:
      os.makedirs(directory)
    except OSError, e:
      if e.errno != os.errno.EEXIST:
        raise


def delete(file_name):
  """
  Delete file if it exists
  :param file_name:
  :return:
  """
  if file_name and file_exists(file_name):
    try:
      shutil.rmtree(file_name)
    except OSError, e:
      if e.errno != os.errno.ENOENT:
        raise


class Buffon(object):
  def __init__(self, r=1.0, l=1):
    self.r = r
    self.l = l
    self.d = l / r
    self.polygon = None
    self.points = None
    self.lines = None
    self.x_limits = None
    self.y_limits = None
    self.initialize()

  def initialize(self):
    raise NotImplementedError("Has to be implemented by subclass")

  def plot_frame(self):
    for line in self.lines:
      x, y = line.xy
      plt.plot(x, y, color="grey", alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)

  @staticmethod
  def plot_point(x, y):
    plt.plot(x, y, 'bo')

  @staticmethod
  def plot_line(line, color="green"):
    x, y = line.xy
    plt.plot(x, y, color=color, alpha=0.7, linewidth=2, solid_capstyle='round', zorder=2)

  def generate(self):
    point = None
    while point is None or not point.intersects(self.polygon):
      x = np.random.uniform(low=self.x_limits[0], high=self.x_limits[1])
      y = np.random.uniform(low=self.y_limits[0], high=self.y_limits[1])
      point = Point(x, y)
    angle = np.random.uniform(0, np.pi)
    hyp = self.l / 2
    dx = hyp * np.cos(angle)
    dy = hyp * np.sin(angle)
    A = (point.x - dx, point.y - dy)
    B = (point.x + dx, point.y + dy)
    line = LineString([A, B])
    return line

  def illustrate(self, n_lines=10, save_file="temp.png"):
    self.initialize()
    self.plot_frame()
    colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow'}
    for _ in range(n_lines):
      line = self.generate()
      color = colors[self.get_intersections(line)]
      self.plot_line(line, color)
    plt.savefig(save_file)

  def get_intersections(self, line):
    cnt = 0
    for border in self.lines:
      if line.intersects(border):
        cnt += 1
    return cnt

  def simulate(self, n):
    self.initialize()
    cuts = {
        0: 0, 1: 0, 2: 0, 3: 0
    }
    for _ in range(n):
      line = self.generate()
      line_cuts = self.get_intersections(line)
      cuts[line_cuts] += 1
    pi_estimate = self.estimate(cuts)
    print(cuts)
    print(pi_estimate)

  def throw(self):
    line = self.generate()
    return self.get_intersections(line)

  def valid_throw(self):
    line = self.generate()
    for border in self.lines:
      if line.intersects(border):
        return 1
    return 0

  def estimate(self, cuts, n=0):
    raise NotImplementedError("Has to be implemented by subclass")


class Buffon3D(Buffon):
  def __init__(self, r=1.0, l=1):
    Buffon.__init__(self, r, l)
    self.alpha = 1.5 * self.r * (4 - np.sqrt(3) * self.r / 2)

  def initialize(self):
    d = self.d
    d_3 = d / np.sqrt(3)
    d2_3 = 2 * d_3
    A = (-d_3, d)
    B = (d_3, d)
    C = (d2_3, 0)
    D = (d_3, -d)
    E = (-d_3, -d)
    F = (-d2_3, 0)
    self.points = [A, B, C, D, E, F]
    self.polygon = Polygon(self.points)
    self.lines = [LineString([A, B]), LineString([B, C]), LineString([C, D]),
                  LineString([D, E]), LineString([E, F]), LineString([F, A]),
                  LineString([F, C]), LineString([A, D]), LineString([B, E])]
    self.x_limits = (-d2_3, d2_3)
    self.y_limits = (-d, d)

  def estimate(self, cuts, n=0):
    p_0 = 0 if n == 0 else cuts[0] / n
    return 2 * self.alpha / (3 - 2 * p_0)


# class Buffon3D(Buffon):
#   def __init__(self, r=1.0, l=1):
#     Buffon.__init__(self, r, l)
#     self.alpha = 1.5 * self.r * (4 - np.sqrt(3) * self.r / 2)
#
#   def initialize(self):
#     d = self.d
#     d_3 = d / np.sqrt(3)
#     d2_3 = 2 * d_3
#     A = (0, 0)
#     B = (0, d2_3)
#     C = (d_3, d)
#     self.points = [A, B, C]
#     self.x_limits = (0, d2_3)
#     self.y_limits = (0, d)
#     self.polygon = Polygon(self.points)
#
#   def get_intersections(self, line):
#     return int(self.polygon.intersects(line))
#
#   def estimate(self, cuts, n=0):
#     r = self.l / self.d
#     # return 3 * r * (8 - np.sqrt(3) * r) / (2 * (3 - 2 * cuts[0] / n))
#     return 2 * self.alpha / (3 - 2 * cuts[0] / n)
#
#   def simulate(self, n):
#     self.initialize()
#     cuts = {
#         0: 0, 1: 0
#     }
#     for _ in range(n):
#       line = self.generate()
#       line_cuts = self.get_intersections(line)
#       cuts[line_cuts] += 1
#     pi_estimate = self.estimate(cuts, n)
#     print(cuts)
#     print(pi_estimate)


class Buffon2D(Buffon):
  def __init__(self, r=1.0, l=1):
    Buffon.__init__(self, r, l)
    self.m = 4 * self.r - self.r ** 2

  def initialize(self):
    d = self.d
    d1_5 = 1.5 * d
    d0_5 = 0.5 * d
    A, B, C, D = (-d1_5, -d), (-d0_5, -d), (d0_5, -d), (d1_5, -d)
    E, F = (-d1_5, 0), (d1_5, 0)
    G, H, I, J = (-d1_5, d), (-d0_5, d), (d0_5, d), (d1_5, d)
    self.points = [A, B, C, D, E, F, G, H, I, J]
    self.polygon = Polygon(self.points)
    self.lines = [LineString([A, D]), LineString([E, F]), LineString([G, J]),
                  LineString([A, G]), LineString([B, H]),
                  LineString([C, I]), LineString([D, J])]
    self.x_limits = (-d1_5, d1_5)
    self.y_limits = (-d, d)

  def estimate(self, cuts, n=0):
    n_throws = sum(cuts.values())
    n_1 = cuts.get(1, 0)
    n_2 = cuts.get(2, 0)
    if n_1 == n_2 == 0:
      return 0
    return self.m * n_throws / (n_1 + n_2)


def pi_needle_triple(r, l, cnt_probe_limit=100000, signif_digits=3, seed=None, reject_censored=False):
  if seed is None:
    seed = np.random.randint(0, 2 ** 16)
  solver = "Needles_3"              # solver name to include in the results
  np.random.seed(seed)
  tolerance = 5 / (10 ** (signif_digits + 1))
  pi_lb = np.pi - tolerance
  pi_ub = np.pi + tolerance
  cnt_probe = 0
  is_censored = True
  experiment = Buffon3D(r, l)
  experiment.initialize()
  cuts = {
      0: 0, 1: 0, 2: 0, 3: 0
  }
  pi_estimate = 0
  start = time.time()
  while cnt_probe < cnt_probe_limit:
    cnt_probe += 1
    n_cuts = experiment.throw()
    cuts[n_cuts] += 1
    pi_estimate = experiment.estimate(cuts)
    if pi_lb <= pi_estimate <= pi_ub:
      is_censored = False
      break
  end = time.time()
  # pi_estimate = round(pi_estimate, signif_digits)
  error = abs(pi_estimate - np.pi)
  if reject_censored and is_censored:
    new_seed = np.random.randint(0, 2 ** 16)
    return pi_needle_triple(r, l, cnt_probe_limit, signif_digits, new_seed, reject_censored)
  else:
    return [
        seed,
        solver,
        signif_digits,
        pi_estimate,
        tolerance,
        error,
        is_censored,
        # "cnt_probe_limit": cnt_probe_limit,
        cnt_probe,
        round(end - start, 4)
    ]


def pi_needle_double(r, l, cnt_probe_limit=100000, signif_digits=3, seed=None, reject_censored=False):
  if seed is None:
    seed = np.random.randint(0, 2 ** 16)
  solver = "Needles_2"  # solver name to include in the results
  np.random.seed(seed)
  tolerance = 5 / (10 ** (signif_digits + 1))
  pi_lb = np.pi - tolerance
  pi_ub = np.pi + tolerance
  cnt_probe = 0
  is_censored = True
  experiment = Buffon2D(r, l)
  experiment.initialize()
  cuts = {
      0: 0, 1: 0, 2: 0, 3: 0
  }
  pi_estimate = 0
  start = time.time()
  while cnt_probe < cnt_probe_limit:
    cnt_probe += 1
    n_cuts = experiment.throw()
    cuts[n_cuts] += 1
    pi_estimate = experiment.estimate(cuts)
    if pi_lb <= pi_estimate <= pi_ub:
      is_censored = False
      break
  end = time.time()
  # pi_estimate = round(pi_estimate, signif_digits)
  error = abs(pi_estimate - np.pi)
  if reject_censored and is_censored:
    new_seed = np.random.randint(0, 2 ** 16)
    return pi_needle_triple(r, l, cnt_probe_limit, signif_digits, new_seed, reject_censored)
  else:
    return [
        seed,
        solver,
        signif_digits,
        pi_estimate,
        tolerance,
        error,
        is_censored,
        # "cnt_probe_limit": cnt_probe_limit,
        cnt_probe,
        round(end - start, 4)
    ]


def _illustrate3d():
  buf = Buffon3D()
  buf.illustrate(10)


def _illustrate2d():
  buf = Buffon2D()
  buf.illustrate(10)


def _pi_needle_triple():
  results = pd.DataFrame(columns=["seedInit", "solverName", "signifDigits", "piHat", "OFtol",
                                  "error", "isCensored", "cntProbe", "runtime"])
  seed = round(9999 * np.random.uniform(0, 1))
  signif_start = 2
  for i in range(1, 100):
    temp_return = pi_needle_triple(1, 1, cnt_probe_limit=1000, signif_digits=signif_start, seed=seed)
    results.loc[i] = temp_return
    seed = round(1e9 * np.random.uniform(0, 1))
    if i % 15 == 0:
      signif_start += 1
  results.index.name = 'sampleId'
  results.columns.name = results.index.name
  results.index.name = None
  results.to_csv("results.txt", sep="\t")
  print(results)
  results.to_html('results.html')
  with open('results.html') as f:
    html_data = f.read()
    html_data += "\n"
  with open('results.html', "wb") as f:
    f.write(html_data)


def plot_errorbar(x, y, e, fig_name, title, y_label):
  plt.errorbar(x, y, e, fmt='bo-', ecolor='red')
  plt.xlabel("# of significant digits")
  plt.ylabel(y_label)
  plt.title(title)
  plt.savefig(fig_name)
  plt.clf()


def plot_line(x, y, fig_name, title, y_label):
  plt.plot(x, y, 'o-')
  plt.xlabel("# of significant digits")
  plt.ylabel(y_label)
  plt.title(title)
  plt.savefig(fig_name)
  plt.clf()


def triple_plot_stats(max_signif_digits, repeats, cnt_probe_limit, start_seed=None, folder_prefix="exp"):
  results = pd.DataFrame(
      columns=["seedInit", "solverName", "signifDigits", "piHat", "OFtol",
               "error", "isCensored", "cntProbe", "runtime"])
  if start_seed is None:
    start_seed = np.random.randint(0, 100)
  np.random.seed(start_seed)
  x_axis = []
  all_trials_means, all_trials_std = [], []
  all_errors_means, all_errors_std = [], []
  i = 1
  for signif_digit in range(1, max_signif_digits + 1):
    x_axis.append(signif_digit)
    trials, errors = [], []
    count_limit = cnt_probe_limit
    iterations = 0
    while iterations < repeats:
      result = pi_needle_triple(1, 1, count_limit, signif_digits=signif_digit)
      if result[-3] is True:
        count_limit *= 2
        iterations = 0
        continue
      error = np.absolute(result[3] - np.pi)
      trials.append(result[-2])
      errors.append(error)
      results.loc[i] = result
      i += 1
      iterations += 1
    errors = np.array(errors).astype(np.float)
    all_trials_means.append(np.mean(trials))
    all_errors_means.append(np.mean(errors))
    all_trials_std.append(np.std(trials))
    all_errors_std.append(np.std(errors))
    print("Completed for %d Significant Digits" % signif_digit)
  results.index.name = 'sampleId'
  results.columns.name = results.index.name
  results.index.name = None
  plot_errorbar(x_axis, all_trials_means, all_trials_std, "%s/trials.png" % folder_prefix,
                "# Trials vs # Significant digits", "#Trials")
  plot_errorbar(x_axis, all_errors_means, all_errors_std, "%s/errors.png" % folder_prefix,
                "Errors vs # Significant digits", "Error")
  plot_line(x_axis, np.log(all_trials_means), "exp/trials_log.png" % folder_prefix,
            "log - # Trials vs # Significant digits", "log - #Trials")
  plot_line(x_axis, np.log(all_errors_means), "exp/errors_log.png" % folder_prefix,
            "log - Error vs # Significant digits", "log - Error")
  results.to_csv("exp/results.txt" % folder_prefix, sep="\t")
  results.to_html('exp/results.html' % folder_prefix)

  with open('exp/results.html' % folder_prefix) as f:
    html_data = f.read()
    html_data += "\n"
  with open('exp/results.html' % folder_prefix, "wb") as f:
    f.write(html_data)


def parallel_3d_throw(*arg, **kwarg):
  return Buffon3D.throw(*arg, **kwarg)


def parallel_2d_throw(*arg, **kwarg):
  return Buffon2D.throw(*arg, **kwarg)


def pi_throws_3d(r, l, throws, num_threads):
  """
  Estimate Pi for certain number of throws for Buffon3D
  :param r: Ratio of length to distance
  :param l: Length of needle
  :param throws: Number of throws
  :param num_threads: Number of threads
  :return: Estimate of Pi
  """
  experiment = Buffon3D(r, l)
  experiment.initialize()
  cuts = {
      0: 0, 1: 0, 2: 0, 3: 0
  }
  results = Parallel(n_jobs=num_threads)(delayed(parallel_3d_throw)(experiment) for _ in range(throws))
  for cut in results:
    cuts[cut] += 1
  pi_estimate = experiment.estimate(cuts, throws)
  return pi_estimate


def pi_throws_2d(r, l, throws, num_threads):
  """
  Estimate Pi for certain number of throws for Buffon 2D
  :param r: Ratio of length to distance
  :param l: Length of needle
  :param throws: Number of throws
  :param num_threads: Number of threads
  :return: Estimate of Pi
  """
  experiment = Buffon2D(r, l)
  experiment.initialize()
  cuts = {
      0: 0, 1: 0, 2: 0, 3: 0
  }
  results = Parallel(n_jobs=num_threads)(delayed(parallel_2d_throw)(experiment) for _ in range(throws))
  for cut in results:
    cuts[cut] += 1
  pi_estimate = experiment.estimate(cuts, throws)
  return pi_estimate


def throws_experiment(r, l, exp_func, throws, repeats, folder="", save_file=None,
                      num_threads=multiprocessing.cpu_count()):
  """
  Perform the thorws experiment where a needle is thrown a certain number of times.
  :param r: Ratio of length to distance
  :param l: Length of needle
  :param exp_func: Throws exp function
  :param throws: Number of throws
  :param repeats: Number of repeats
  :param folder: Folder to save the results
  :param save_file: Save file. If None prints to console
  :param num_threads: Number of threads to run in parallel
  :return:
  """
  ret_vals, seeds, runtimes = [], [], []
  print("Running on %d threads" % num_threads)
  for _ in range(repeats):
    seed = np.random.randint(0, 2 ** 16)
    np.random.seed(seed)
    seeds.append(seed)
    start = time.time()
    ret_vals.append(exp_func(r, l, throws, num_threads))
    runtimes.append(time.time() - start)
  mkdir(folder)
  if save_file is not None:
    pi_file = "%s%s" % (folder, save_file)
    mode = "ab" if file_exists(pi_file) else "wb"
    with open(pi_file, mode) as f:
      f.write("\n".join(map(str, ret_vals)))
    seeds_file = "%sseeds_%s" % (folder, save_file)
    mode = "ab" if file_exists(seeds_file) else "wb"
    with open(seeds_file, mode) as f:
      f.write("\n".join(map(str, seeds)))
    times_file = "%stimes_%s" % (folder, save_file)
    mode = "ab" if file_exists(times_file) else "wb"
    with open(times_file, mode) as f:
      f.write("\n".join(map(str, runtimes)))
  return ret_vals


def read_throw_results(file_name):
  results = []
  with open(file_name) as f:
    for line in f.readlines():
      results.append(float(line))
  return results


def compare_mathematica_python(x_axis, fig_name):
  mathematica_means, mathematica_std = [], []
  python_means, python_std = [], []
  for x in x_axis:
    mathematica_file = "results-mathematica/triplegrid_%d.csv" % x
    mathematica_results = read_throw_results(mathematica_file)
    python_file = "results-python/triplegrid_%d.csv" % x
    python_results = read_throw_results(python_file)
    mathematica_means.append(np.mean(mathematica_results))
    mathematica_std.append(np.std(mathematica_results))
    python_means.append(np.mean(python_results))
    python_std.append(np.std(python_results))
  python_error_means = abs(np.subtract([np.pi] * len(python_means), python_means))
  mathematica_error_means = abs(np.subtract([np.pi] * len(mathematica_means), mathematica_means))
  # plt.errorbar(x_axis, mathematica_means, mathematica_std, fmt='bo-', ecolor='red')
  # plt.errorbar(x_axis, python_means, python_std, fmt='g+-', ecolor='yellow')
  plt.plot(x_axis, mathematica_error_means, 'bo-', label="Mathematica")
  plt.plot(x_axis, python_error_means, 'g+-', label="Python")
  print("# Mathematica")
  print(mathematica_error_means)
  print("# Python")
  print(python_error_means)
  # plt.plot(x_axis, mathematica_means, 'bo-', label="Mathematica")
  # plt.plot(x_axis, python_means, 'g+-', label="Python")
  plt.legend()
  # plt.ylabel("Estimate of Pi")
  plt.ylabel("Error of Pi")
  plt.xlabel("Number of throws")
  plt.xscale("log")
  plt.yscale("log")
  plt.title("Comparing Mathematica and Python")
  plt.savefig(fig_name)
  plt.clf()


def _compare_mathematica_python():
  compare_mathematica_python([10, 100, 1000, 10000], "fg_asym_pi_triplegrid_mathematica_vs_python.png")


def aggregate_throws(throws, folder, write_file):
  print("## Python")
  with open(write_file, "wb") as csv_file:
    writer = csv.writer(csv_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    writer.writerow(['sampleId', 'seedInit', 'solverName', 'numThrows', 'piHat', 'error', 'runtime'])
    sampleId = 1
    for throw in throws:
      pi_file = open("%s/triplegrid_%d.csv" % (folder, throw))
      seed_file = open("%s/seeds_triplegrid_%d.csv" % (folder, throw))
      time_file = open("%s/times_triplegrid_%d.csv" % (folder, throw))
      estimates = []
      for pi_est, seed, runtime in zip(pi_file.readlines(), seed_file.readlines(), time_file.readlines()):
        pi_est = float(pi_est)
        writer.writerow([sampleId, int(seed), 'needles3', throw, pi_est, np.abs(np.pi - pi_est), float(runtime)])
        sampleId += 1
        estimates.append(pi_est)
      print(throw, abs(np.pi - np.mean(estimates)))


def aggregate_mathematica(throws, folder, write_file):
  print("## Mathematica")
  with open(write_file, "wb") as csv_file:
    writer = csv.writer(csv_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    writer.writerow(['sampleId', 'numThrows', 'piHat', 'error'])
    sampleId = 1
    for throw in throws:
      pi_file = open("%s/triplegrid_%d.csv" % (folder, throw))
      estimates = []
      for pi_est in pi_file.readlines():
        pi_est = float(pi_est)
        writer.writerow([sampleId, throw, pi_est, np.abs(np.pi - pi_est)])
        sampleId += 1
        estimates.append(pi_est)
      print(throw, abs(np.pi - np.mean(estimates)))


def _aggregate_throws():
  aggregate_throws([10, 100, 1000, 10000, 100000, 1000000], "results-python", "results-python/aggregate.txt")
  aggregate_mathematica([10, 100, 1000, 10000], "results-mathematica", "results-mathematica/aggregate.txt")


def _triple_plot_stats():
  # triple_plot_stats(6, 100, 100000)
  triple_plot_stats(2, 5, 100000)

# if __name__ == "__main__":
#   # _pi_needle_triple()
#   # _illustrate()
#   # _triple_plot_stats()
#   # _compare_mathematica_python()
#   _throws_experiments()


def validate_sd():
  csv_file = "results-python/needles3/fg_asym_pi_plain_needles3_4_56296.txt"
  with open(csv_file) as f:
    reader = csv.DictReader(f, delimiter='\t')
    throws = collections.defaultdict(list)
    for row in reader:
      throw = int(row['numThrows'])
      pi_hat = float(row['piHat'])
      throws[throw].append(pi_hat)
    std_devs = []
    std_dev_theoritical = []
    for throw in sorted(throws.keys()):
      std_devs.append(np.std(throws[throw]))
      # std_dev_theoritical.append(np.sqrt(0.01578/throw))
      std_dev_theoritical.append(np.sqrt(0.01578 / throw))
  print(std_devs)
  print(std_dev_theoritical)

def plot_sd(file_name, fig_name="temp.png"):
  with open(file_name) as f:
    reader = csv.DictReader(f, delimiter='\t')
    throws = collections.defaultdict(list)
    for row in reader:
      throw = int(row['numThrows'])
      pi_hat = float(row['piHat'])
      throws[throw].append(pi_hat)
    std_devs = []
    std_dev_theoritical = []
    x_axis = []
    for throw in sorted(throws.keys()):
      std_devs.append(np.std(throws[throw]))
      std_dev_theoritical.append(np.sqrt(0.01578 / throw))
      x_axis.append(throw)
  plt.plot(x_axis, std_devs, color='r', linestyle='-', marker='+', label="Experimental")
  plt.plot(x_axis, std_dev_theoritical, color='g', linestyle='--', marker='o', label="Theoretical")
  plt.savefig(fig_name)
  plt.clf()

# plot_sd("results-python/needles3/fg_asym_pi_plain_needles3_4_56296.txt")

# validate_sd()
