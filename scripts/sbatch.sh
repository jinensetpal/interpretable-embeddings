#!/usr/bin/bash

#SBATCH --account=gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 6
#SBATCH --constraint=V100_32GB
#SBATCH --time=1:15:00

module load cuda cudnn anaconda
source activate dhruv

cd ~/interpretable-embeddings

python -m src.evaluatation.evaluate