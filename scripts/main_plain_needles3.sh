#!/bin/bash

#SBATCH --job-name b3dPlain
#SBATCH -N 1
#SBATCH -p opteron
# Use modules to set the software environment

python main_plain_needles3.py -p 6