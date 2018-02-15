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
    self.alpha = 1.5 * self.r * (4 - np.sqrt(3) * self.r / 2)
    self.initialize()

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


  def plot_frame(self):
    for line in self.lines:
      x, y = line.xy
      plt.plot(x, y, color="grey", alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)

  def plot_point(self, x, y):
    plt.plot(x, y, 'bo')

  def plot_line(self, line, color="green"):
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
    colors = {0: 'red', 1:'blue', 2:'green', 3:'yellow'}
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
    for _ in xrange(n):
      line = self.generate()
      line_cuts = self.get_intersections(line)
      cuts[line_cuts] += 1
    pi_estimate = self.estimate(cuts)
    print(cuts)
    print(pi_estimate)

  def throw(self):
    line = self.generate()
    return self.get_intersections(line)

  def estimate(self, cuts):
    n_throws = sum(cuts.values())
    return self.alpha / ((cuts[1] + cuts[2] + cuts[3]) / n_throws + 0.5)


def pi_needle_triple(r, l, cnt_probe_limit=10000, signif_digits=3, seed=None):
  if seed is None:
    seed = np.random.randint(0)
  np.random.seed(seed)
  tolerance = 5 / (10 ** (signif_digits + 1))
  pi_lb = np.pi - tolerance
  pi_ub = np.pi + tolerance
  cnt_probe = 0
  is_censored = True
  experiment = Buffon(r, l)
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
  pi_estimate = round(pi_estimate, signif_digits)
  error = ('%'+'.%de' % signif_digits)%(pi_estimate - np.pi)
  return {
    "seed": seed,
    "pi_estimate": pi_estimate,
    "signif_digits": signif_digits,
    "tolerance": tolerance,
    "error": error,
    "is_censored": is_censored,
    "cnt_probe_limit": cnt_probe_limit,
    "cnt_probe": cnt_probe,
    "time(secs)": round(end - start, 4)
  }


def _illustrate():
  buf = Buffon()
  buf.illustrate(10)


def _pi_needle_triple():
  for i in range(5):
    results = pi_needle_triple(1, 1, cnt_probe_limit=100000, signif_digits=4, seed=i)
    print(results)



if __name__ == "__main__":
  _pi_needle_triple()
  # _illustrate()

