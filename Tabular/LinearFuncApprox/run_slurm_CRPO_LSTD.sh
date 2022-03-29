#!/bin/bash
mkdir -p ./out
mkdir -p ./err
Lr_p=(0.75)
iters=1000
samples=(1000)
tiling_sizes=(14)
eta=0
jobname='CRPO'
for tiling_size in "${tiling_sizes[@]}"
do
  for sample in "${samples[@]}"
  do
    for lrp in "${Lr_p[@]}"
    do
      for run in {1..1}
      do
        sbatch -J $jobname ./run_CRPO_tc.sh $lrp $iters $run $sample $tiling_size $eta
      done
    done
  done
done
