#!/bin/bash

#SBATCH --job-name=vae_best
#SBATCH --partition=a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --output=./slurm/vae-%A.log
#SBATCH --error=./slurm/vae-%A.log

source /dssg/home/acct-stu/stu491/.bashrc
conda activate sed

python train.py --encoder MLP3 --decoder MLP3 --z_dim 16 --no_tqdm
