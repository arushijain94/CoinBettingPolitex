#!/bin/bash
mkdir -p ./out
mkdir -p ./err

Lr_p=(1.0)
Lr_l=(1.0)
iters=400
samples=(300)
gammas=(0.9)
b_thresh=1.5
tiling_sizes=(14)
jobname='GDA'

for tiling_size in "${tiling_sizes[@]}"
do
  for gam in "${gammas[@]}"
  do
    for sample in "${samples[@]}"
    do
      for lrp in "${Lr_p[@]}"
      do
        for lrl in "${Lr_l[@]}"
        do
          for run in {1..1}
          do
            sbatch -J $jobname ./run_GDA_KW.sh $lrp $lrl $iters $run $sample $gam $b_thresh $tiling_size
          done
        done
      done
    done
  done
done
