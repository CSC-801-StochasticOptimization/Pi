#!/bin/bash

#SBATCH --job-name b3dPlain
#SBATCH -N 1
#SBATCH -p opteron
#SBATCH -x c[79-98,101-107]
# Use modules to set the software environment

python main_plain_needles3.py -p 6 -n 16 -r 100 -f main
