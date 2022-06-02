#!/bin/bash

#SBATCH --job-name=avsc_best
#SBATCH -p a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

source /dssg/home/acct-stu/stu491/.bashrc
conda activate sed

python train.py --config_file configs/late_weighted.yaml
python evaluate.py --experiment_path experiments/late_weighted
python eval_prediction.py --prediction ./experiments/late_weighted/prediction.csv --label ./data/evaluate/fold1_evaluate.csv
