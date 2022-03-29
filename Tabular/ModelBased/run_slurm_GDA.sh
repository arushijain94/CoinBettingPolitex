#!/bin/bash
mkdir -p ./out
mkdir -p ./err

Lr_p=(1.0)
Lr_l=(0.1)
avg_window=200
full_average=(1)
multiple_constraints=(0)
iters=1500
jobname='GDA'
gammas=(0.9)

for gam in "${gammas[@]}"
do
  for lrp in "${Lr_p[@]}"
  do
    for lrl in "${Lr_l[@]}"
    do
      for favg in "${full_average[@]}"
      do
        for mc in "${multiple_constraints[@]}"
        do
          for run in {1..5}
          do
            sbatch -J $jobname ./run_GDA.sh $lrp $lrl $iters $mc $favg $avg_window $run $gam
          done
        done
      done
    done
  done
done
