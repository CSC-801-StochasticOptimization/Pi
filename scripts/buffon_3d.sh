#!/bin/bash

#SBATCH --job-name buffon3d
#SBATCH -N 1
#SBATCH -p opteron
# Use modules to set the software environment

python buffon_3d.py