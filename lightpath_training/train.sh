#!/bin/bash
#SBATCH --job-name=train_l
#SBATCH --ntasks=4
#SBATCH --mem=40000
#SBATCH --time=48:00:00
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=santiolmedo99@gmail.com


source ~/miniconda3/bin/activate
conda activate gnn_env

python -m lightpath_training.train