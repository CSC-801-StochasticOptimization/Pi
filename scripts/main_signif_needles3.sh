#!/bin/bash

#SBATCH --job-name b3dSignif
#SBATCH -N 1
#SBATCH -p opteron
# Use modules to set the software environment


python main_signif_needles3.py -d 7 -p 100 -s $1