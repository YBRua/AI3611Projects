#!/bin/bash

#SBATCH --job-name=vae_best
#SBATCH --partition=a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --output=./slurm/lm-%A.log
#SBATCH --error=./slurm/lm-%A.log

source /dssg/home/acct-stu/stu463/.bashrc
conda activate espnet

python train.py --encoder MLP3 --decoder MLP3 --zdim 16
