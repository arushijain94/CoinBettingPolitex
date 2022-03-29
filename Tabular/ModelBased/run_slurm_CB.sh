#!/bin/bash
mkdir -p ./out
mkdir -p ./err

avg_window=200
full_average=(1)
multiple_constraints=(0)
iters=1500
jobname='CB'
alpha_lambd=(8)
gammas=(0.9)

for gam in "${gammas[@]}"
do
  for alpha in "${alpha_lambd[@]}"
  do
    for favg in "${full_average[@]}"
    do
      for mc in "${multiple_constraints[@]}"
      do
        for run in {1..5}
        do
          sbatch -J $jobname ./run_CB.sh $iters $mc $avg_window $favg $run $alpha $gam
        done
      done
    done
  done
done
