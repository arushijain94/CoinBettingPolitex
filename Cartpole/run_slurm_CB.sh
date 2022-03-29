#!/bin/bash
mkdir -p ./out
mkdir -p ./err
iters=200
jobname='CB'
alpha_lambd=(0.1 5 50 250 500 750 1000)
sample=50
entrop_coeffs=(0.001 0.01 0.1 0.0)
cmdp=1

for entropy in "${entrop_coeffs[@]}"
do
  for alpha in "${alpha_lambd[@]}"
  do
    for run in {1..5}
    do
      sbatch -J $jobname ./run_CB.sh $iters $run $alpha $sample $entropy $cmdp
    done
  done
done
