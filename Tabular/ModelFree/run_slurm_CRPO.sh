#!/bin/bash
mkdir -p ./out
mkdir -p ./err

iters=1500
jobname='CRPO'
eta=0.0
lrps=(0.75)
gammas=(0.9)
samples=(2000 3000 5000)

for sample in "${samples[@]}"
do
  for gam in "${gammas[@]}"
  do
    for lrp in "${lrps[@]}"
    do
      for run in {1..5}
      do
        sbatch -J $jobname ./run_CRPO.sh $iters $run $eta $lrp $gam $sample
      done
    done
  done
done
