#!/bin/bash

#SBATCH --job-name buffon3d
#SBATCH -N 1
#SBATCH -p opteron
# Use modules to set the software environment

python main_plain_needles3.py -d 6 -p 100