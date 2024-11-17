#!/bin/bash
#SBATCH --job-name=train_l
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=49152
#SBATCH --time=115:00:00
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=santiolmedo99@gmail.com


source ~/miniconda3/bin/activate
conda activate gnn_env

python -m lightpath_training.train