source /dssg/home/acct-stu/stu463/.bashrc
conda activate espnet

python train_with_optim.py --data data/gigaspeech --cuda --epochs 20 --model LSTM --lr 0.001 --emsize 512 --nhid 512 --no_tqdm
