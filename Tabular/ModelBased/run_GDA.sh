#!/bin/bash
#SBATCH --mem=100Mb	                      # Ask for 2 GB of RAM
#SBATCH --gres=gpu:1             # Number of GPUs (per node)
#SBATCH --cpus-per-task=1             # Number of CPUs (per node)
#SBATCH --time=0:10:00                   # The job will run for 3 hours
#SBATCH --output=./out/%j-%x.out
#SBATCH --error=./err/%j-%x.err

module load python/3.7
source ~/.bashrc
source ~/CMDPenv/bin/activate

lrp=$1
lrl_val=$2
iters=$3
mc=$4
favg=$5
avg_window=$6
run=$7
gam=$8
path_name="./"
file_name="GDA.py"

python $path_name$file_name --learning_rate_pol $lrp --learning_rate_lambd $lrl_val --num_iterations $iters --multiple_constraints $mc --full_average $favg --moving_avg_window $avg_window --run $run --gamma $gam
