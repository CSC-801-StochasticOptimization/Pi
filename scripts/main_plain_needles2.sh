#!/bin/bash

#SBATCH --job-name b2dPlain
#SBATCH -N 1
#SBATCH -p opteron
# Use modules to set the software environment

python main_plain_needles2.py -p 6