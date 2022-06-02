#!/bin/bash

#SBATCH --job-name=lm_best
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

python train_with_optim.py --data data/gigaspeech --cuda --epochs 20 --model LSTM --lr 0.001 --emsize 512 --nhid 512 --no_tqdm
