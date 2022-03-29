#!/bin/bash
#SBATCH --mem=600M	                      # Ask for 2 GB of RAM
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --time=10:00:00                   # The job will run for 3 hours
#SBATCH --output=./out/%j-%x.out
#SBATCH --error=./err/%j-%x.err

module load python/3.7
source ~/.bashrc
source ~/CMDPenv/bin/activate

lrp=$1
iters=$2
run=$3
sample=$4
tiling_size=$5
eta=$6
path_name="./"
file_name="CRPO_LSTD.py"

python $path_name$file_name --learning_rate $lrp --num_iterations $iters --run $run --num_samples $sample --tiling_size $tiling_size --eta $eta
