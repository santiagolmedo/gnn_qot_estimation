#!/bin/bash
#SBATCH --job-name=test_l
#SBATCH --ntasks=1
#SBATCH --mem=100000
#SBATCH --time=120:00:00
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=santiolmedo99@gmail.com

source ~/miniconda3/bin/activate
conda activate gnn_env

python -m lightpath_training.test