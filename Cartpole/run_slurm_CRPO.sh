#!/bin/bash
mkdir -p ./out
mkdir -p ./err
iters=200
jobname='CRPO'
Lr_p=(0.1 0.5 0.01 0.001 0.005 0.0001 0.0005)
etas=(0 10)
samples=(50)
entrop_coeffs=(0.001 0.01 0.1 0.0)

for eta in "${etas[@]}"
do
  for entropy in "${entrop_coeffs[@]}"
  do
    for sample in "${samples[@]}"
    do
      for lrp in "${Lr_p[@]}"
      do
        for run in {1..5}
        do
          sbatch -J $jobname ./run_CRPO.sh $iters $run $lrp $sample $entropy $eta
        done
      done
    done
  done
done
