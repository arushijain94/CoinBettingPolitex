#!/bin/bash
mkdir -p ./out
mkdir -p ./err

avg_window=200
full_average=(1)
multiple_constraints=(0)
iters=1500
jobname='CRPO'
eta=0.25
lrps=(0.001 0.01 0.05 0.1 0.5 0.75)
gammas=(0.9)

for gam in "${gammas[@]}"
do
  for lrp in "${lrps[@]}"
  do
    for favg in "${full_average[@]}"
    do
      for mc in "${multiple_constraints[@]}"
      do
        for run in {1..5}
        do
          sbatch -J $jobname ./run_CRPO.sh $iters $mc $avg_window $favg $run $eta $lrp $gam $b_thresh
        done
      done
    done
  done
done
