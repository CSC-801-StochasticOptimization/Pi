#!/bin/bash

for number in {1..5}
do
  sbatch scripts/main_signif_needles3.sh ${number}
done