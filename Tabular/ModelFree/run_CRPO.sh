#!/bin/bash
#SBATCH --mem=800M	                      # Ask for 2 GB of RAM
#SBATCH --gres=gpu:1             # Number of GPUs (per node)
#SBATCH --cpus-per-task=1
#SBATCH --time=15:00:00                   # The job will run for 3 hours
#SBATCH --output=./out/%j-%x.out
#SBATCH --error=./err/%j-%x.err

module load python/3.7
source ~/.bashrc
source ~/CMDPenv/bin/activate

iters=$1
run=$2
eta=$3
lrp=$4
gam=$5
nsamples=$6
path_name="./"
file_name="CRPO_TDSampling.py"

python $path_name$file_name --num_iterations $iters --run $run --eta $eta --lrp $lrp --gamma $gam --num_samples $nsamples
