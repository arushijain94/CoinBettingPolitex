#!/bin/bash
#SBATCH --mem=100Mb	                      # Ask for 2 GB of RAM
#SBATCH --gres=gpu:1             # Number of GPUs (per node)
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00                   # The job will run for 3 hours
#SBATCH --output=./out/%j-%x.out
#SBATCH --error=./err/%j-%x.err

module load python/3.7
source ~/.bashrc
source ~/CMDPenv/bin/activate

iters=$1
mc=$2
avg_window=$3
favg=$4
run=$5
alpha=$6
gam=$7
path_name="./"
file_name="CB.py"

python $path_name$file_name --num_iterations $iters --multiple_constraints $mc --moving_avg_window $avg_window --full_average $favg --run $run --alpha_lambd $alpha --gamma $gam
