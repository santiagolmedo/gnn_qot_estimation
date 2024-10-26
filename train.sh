#!/bin/bash
#SBATCH --job-name=train
#SBATCH --ntasks=1
#SBATCH --mem=64000
#SBATCH --time=48:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:p100:2
#SBATCH --qos=gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=santiolmedo99@gmail.com


source ~/miniconda3/bin/activate
conda activate antel

python train.py