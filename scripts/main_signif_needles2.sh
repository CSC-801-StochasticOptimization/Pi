#!/bin/bash

#SBATCH --job-name b2dSignif
#SBATCH -N 1
#SBATCH -p opteron
# Use modules to set the software environment

python main_signif_needles2.py -d 6 -p 100