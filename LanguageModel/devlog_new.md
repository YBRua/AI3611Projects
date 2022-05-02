# Devlog Reborn

## Baselines

### LSTM-200

```sh
python main.py --data data/gigaspeech --cuda --epochs 6 --model LSTM --lr 20 --emsize 200 --nhid 200 --nlayers 2
```

- test loss  5.11 
- test ppl   164.97
- 2022-04-27-11-03-17-LSTM-200

### GRU-200

```sh
python main.py --data data/gigaspeech --cuda --epochs 6 --model GRU --lr 20 --emsize 200 --nhid 200 --nlayers 2
```

- test loss 5.67
- test ppl 290.05
- 2022-04-27-11-11-36-GRU-200

### RNN-RELU-200

```sh
python main.py --data data/gigaspeech --cuda --epochs 6 --model RNN_RELU --lr 20 --emsize 200 --nhid 200 --nlayers 2
```

- NaN
- Train failed

### RNN-TANH-200

```sh
python main.py --data data/gigaspeech --cuda --epochs 6 --model RNN_TANH --lr 20 --emsize 200 --nhid 200 --nlayers 2
```

- test loss  6.24
- test ppl   514.92
- 2022-04-27-12-36-52-RNN_TANH

## Optimizer Ready

### RNN-RELU-200

```sh
python train_with_optim.py --data data/gigaspeech --cuda --epochs 6 --model RNN_RELU --lr 0.001 --emsize 200 --nhid 200 --no_tqdm
```

- | Test loss  5.412 | Test Ppl    223.97 |
- 2022-04-27-15-14-59-RNN_RELU

### RNN-TANH-200

```sh
python train_with_optim.py --data data/gigaspeech --cuda --epochs 6 --model RNN_TANH --lr 0.001 --emsize 200 --nhid 200 --no_tqdm
```

- | Test loss  5.469 | Test Ppl    237.12 |
- 2022-04-27-14-23-00-RNN_TANH

### LSTM-200

```sh
python train_with_optim.py --data data/gigaspeech --cuda --epochs 6 --model LSTM --lr 0.001 --emsize 200 --nhid 200 --no_tqdm
```

- Test loss  5.092
- Test Ppl    162.71
- 2022-04-27-11-33-10-LSTM

#### LR 0.0005

[INFO 2022-04-27 12:52:25,396]: | Test loss  5.115 | Test Ppl    166.50 |

#### LR 0.005

| Test loss  5.200 | Test Ppl    181.31 |

#### Epoch 10

> Best from E10

- | Test loss  5.056 | Test Ppl    157.03 |
- 2022-04-27-15-34-45-LSTM-200-10

### GRU-200

```sh
python train_with_optim.py --data data/gigaspeech --cuda --epochs 6 --model GRU --lr 0.001 --emsize 200 --nhid 200 --no_tqdm
```

| Test loss  5.041 | Test Ppl    154.61 |

#### LR 0.0005

- | Test loss  5.043 | Test Ppl    154.97 |
- 2022-04-27-13-03-22-GRU

#### LR 0.002

- | Test loss  5.149 | Test Ppl    172.29 |
- 2022-04-27-15-15-16-GRU

#### 15 Epoch

> Best from E10

- | Test loss  5.025 | Test Ppl    152.21 |
- 2022-04-27-15-34-50-GRU-200-15

### LSTM-512

```sh
train_with_optim.py --data data/gigaspeech --cuda --epochs 6 --model LSTM --lr 0.001 --emsize 512 --nhid 512 --no_tqdm
```

- | Test loss  4.934 | Test Ppl    138.92 |
- 2022-04-27-12-56-43-LSTM

### GRU-512

```sh
train_with_optim.py --data data/gigaspeech --cuda --epochs 6 --model GRU --lr 0.001 --emsize 512 --nhid 512 --no_tqdm
```

- | Test loss  4.993 | Test Ppl    147.45 |
- 2022-04-27-14-22-46-GRU
