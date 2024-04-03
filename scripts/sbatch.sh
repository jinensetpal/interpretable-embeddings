#!/usr/bin/bash

#SBATCH --account=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=V100_32GB
#SBATCH --time=3:30:00

module load cuda cudnn anaconda
source activate dhruv

cd ~/interpretable-embeddings

python -m src.evaluatation.evaluate
