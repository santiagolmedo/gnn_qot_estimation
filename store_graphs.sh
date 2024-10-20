#!/bin/bash
#SBATCH --job-name=store_graphs
#SBATCH --ntasks=1
#SBATCH --mem=100000
#SBATCH --time=120:00:00
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=santiolmedo99@gmail.com

source ~/miniconda3/bin/activate
conda activate antel

python store_graphs.py --dataset_type full --representation topological
python store_graphs.py --dataset_type full --representation lightpath