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
lrl_val=$2
iters=$3
run=$4
sample=$5
gam=$6
b_thresh=$7
tiling_size=$8
path_name="./"
file_name="GDA_LSTD.py"

python $path_name$file_name --learning_rate_pol $lrp --learning_rate_lambd $lrl_val --num_iterations $iters --run $run --num_samples $sample --gamma $gam --b_thresh $b_thresh --tiling_size $tiling_size
