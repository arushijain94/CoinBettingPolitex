#!/bin/bash
mkdir -p ./out
mkdir -p ./err
iters=200
jobname='GDA'
Lr_p=(0.1 0.01 0.001 0.0001)
Lr_l=(0.1 0.01 0.001 0.0001)
samples=(50)
entrop_coeffs=(0.001 0.01 0.1 0.0)
cmdp=1
for entropy in "${entrop_coeffs[@]}"
do
  for sample in "${samples[@]}"
  do
    for lrp in "${Lr_p[@]}"
    do
      for lrl in "${Lr_l[@]}"
      do
        for run in {1..5}
        do
          sbatch -J $jobname ./run_GDA.sh $iters $run $lrp $lrl $sample $entropy $cmdp
        done
      done
    done
  done
done
