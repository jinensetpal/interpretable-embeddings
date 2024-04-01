#!/usr/bin/bash

#SBATCH --account=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=V100_32GB
#SBATCH --time=3:30:00

module load cuda cudnn anaconda
source activate venv

cd ~/interpretable-embeddings

# MLFLOW_TRACKING_USERNAME=jinensetpal \
# MLFLOW_TRACKING_PASSWORD=$MLFLOW_TOKEN \
python -m src.pca-pickle
