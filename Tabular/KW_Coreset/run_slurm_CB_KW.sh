#!/bin/bash
mkdir -p ./out
mkdir -p ./err
iters=400
jobname='C'
alpha_lambd=(0.25)
samples=(300)
gammas=(0.9)
b_thresh=1.5
tiling_sizes=(14)

for tiling_size in "${tiling_sizes[@]}"
do
  for gam in "${gammas[@]}"
  do
    for sample in "${samples[@]}"
    do
      for alpha in "${alpha_lambd[@]}"
      do
        for run in {1..1}
        do
          sbatch -J $jobname ./run_CB_KW.sh $iters $run $alpha $sample $gam $b_thresh $tiling_size
        done
      done
    done
  done
done
