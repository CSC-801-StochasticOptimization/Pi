#!/bin/bash

for number in {1..20}
do
  sbatch scripts/main_plain_needles3_single.sh
done
