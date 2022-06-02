source /dssg/home/acct-stu/stu491/.bashrc
conda activate sed

python train.py --config_file configs/$1.yaml
python evaluate.py --experiment_path experiments/$1
python eval_prediction.py --prediction ./experiments/$1/prediction.csv --label ./data/evaluate/fold1_evaluate.csv
