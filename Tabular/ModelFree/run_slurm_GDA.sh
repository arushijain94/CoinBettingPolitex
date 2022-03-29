#!/bin/bash
mkdir -p ./out
mkdir -p ./err

Lr_p=(1.0)
Lr_l=(0.1)
iters=500
samples=(2000)
gammas=(0.9)
b_vals=(1.5)
jobname='GDA_Sam'

for b_thresh in "${b_vals[@]}"
do
  for gam in "${gammas[@]}"
  do
    for sample in "${samples[@]}"
    do
      for lrp in "${Lr_p[@]}"
      do
        for lrl in "${Lr_l[@]}"
        do
          for run in {2..5}
          do
  #          lrl_val="$(echo "scale=10;$lrp*$lrl" | bc)"
            sbatch -J $jobname ./run_GDA.sh $lrp $lrl $iters $run $sample $gam $b_thresh
          done
        done
      done
    done
  done
done
#done
