#!/bin/bash

#SBATCH -p a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

source /dssg/home/acct-stu/stu491/.bashrc
conda activate sed

# training
python main.py train_evaluate --config_file configs/resnet101_attention.yaml
