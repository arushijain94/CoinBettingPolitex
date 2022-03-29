#!/bin/bash
mkdir -p ./out
mkdir -p ./err

iters=1000
jobname='CB_Hyper'
alpha_lambd=(1 2 5)
samples=(2000)
gammas=(0.9)
b_thresh=1.5

for gam in "${gammas[@]}"
do
  for sample in "${samples[@]}"
  do
    for alpha in "${alpha_lambd[@]}"
    do
      for run in {1..5}
      do
        sbatch -J $jobname ./run_CB.sh $iters $run $alpha $sample $gam $b_thresh
      done
    done
  done
done
