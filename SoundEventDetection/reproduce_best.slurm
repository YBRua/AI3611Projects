#!/bin/bash

#SBATCH --job-name=sed_best
#SBATCH -p a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

source /dssg/home/acct-stu/stu491/.bashrc
conda activate sed

python run.py train_evaluate configs/res_xblk_d0.2.yaml data/eval/feature.csv data/eval/label.csv --window_size=25
