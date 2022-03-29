#!/bin/bash
#SBATCH --mem=800M	                      # Ask for 2 GB of RAM
#SBATCH --gres=gpu:1             # Number of GPUs (per node)
#SBATCH --time=7:00:00                   # The job will run for 3 hours
#SBATCH --output=./out/%j-%x.out
#SBATCH --error=./err/%j-%x.err

module load python/3.7
source ~/.bashrc
source ~/CMDPenv/bin/activate

iters=$1
run=$2
alpha=$3
samples=$4
gam=$5
b_thresh=$6
path_name="./"
file_name="CBP_TDSampling.py"

python $path_name$file_name --num_iterations $iters --run $run --alpha_lambd $alpha --num_samples $samples --gamma $gam --b_thresh $b_thresh
