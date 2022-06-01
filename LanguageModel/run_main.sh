source /dssg/home/acct-stu/stu463/.bashrc
conda activate espnet

python main.py --data data/gigaspeech --cuda --epochs 6 --model LSTM --lr 20 --emsize 200 --nhid 200 --nlayers 2
