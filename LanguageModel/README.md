# AI3611 Project Language Model

## 快速开始

### 交我算超算平台

本项目代码位于 `/dssg/home/acct-stu/stu491/projects/LanguageModel` 目录下. 我们使用助教提供的 Conda 环境 (`espnet`) 进行实验. 

### Shell 和 Sbatch 脚本

#### 复现最优结果

> 最优结果为 Adam 训练的 BiLSTM 模型，Hidden size 512, Embedding size 512.

- 要一键复现最优结果，请运行

```sh
sbatch reproduce_best.slurm
```

- 或者在 `srun` 申请到 GPU 节点后运行 `reproduce_best.slurm` 中的代码，使用 `start_srun.sh` 可以一键申请 GPU 节点. 

```sh
source start_srun.sh
```

获得 GPU 资源后，运行

```sh
source /dssg/home/acct-stu/stu463/.bashrc
conda activate espnet

python train_with_optim.py --data data/gigaspeech --cuda --epochs 20 --model LSTM --lr 0.001 --emsize 512 --nhid 512 --no_tqdm
```

或者，获得 GPU 资源后，直接运行 `run_adam.sh`

```sh
source run_adam.sh
```

#### 基线代码

获得 GPU 资源后，运行 `run_main.sh` 可以运行基线代码

```sh
source run_main.sh
```

### Python 代码

本项目包含两个训练/测试脚本. 

- `main.py`: 基线默认训练脚本. 
- `train_with_optim.py`: 改进后的训练脚本. 重写了训练流程，使用了 Adam 优化器进行优化.

#### 命令行参数

两个脚本共用命令行参数

- `--data`: 数据集目录. 交我算平台上的数据集通过符号链接链接到助教账号下的数据集. 通常为 `./data/gigaspeech`.
- `--model`: 模型架构. 可选参数为 `RNN_TANH`, `RNN_RELU`, `LSTM`, `GRU`, `Transformer`. 默认为 `LSTM`
- `--emsize`: 词嵌入的维度大小. 默认为 200.
- `--nhid`: RNN 或 Transformer 的隐藏层大小. 默认为 200.
- `--nlayers`: RNN 或 Transformer 的层数. 默认为 2.
- `--encoder_layers`: Transformer 专用. 编码器层数. 默认为 2.
- `--decoder_layers`: Transformer 专用. 解码器层数. 默认为 2.
- `--nhead`: Transformer 专用. 多头自注意力的注意力头个数. 默认为 2.
- `--lr`: 学习率. 默认为 20. 使用 `train_with_optim.py` 时需要调整，建议 `0.001`
- `--clip`: 默认脚本专用. 梯度截断.
- `--epochs`: 训练轮数. 默认 6.
- `--batch_size`: Batch Size. 默认 20.
- `--bptt`: 序列长度. 默认 35.
- `--dropout`: Dropout Rate. 默认 0.2.
- `--cuda`: 是否使用 GPU 训练.
- `--save`: 模型保存路径.
- `--dry-run`: 跳过训练，仅运行测试.
- `--no_tqdm`: `train_with_optim` 专用. 不使用 tqdm (不显示进度条)

#### 运行示例

基线代码训练 Transformer

```sh
python main.py --data data/gigaspeech --cuda --epochs 6 --model Transformer --lr 20 --emsize 200 --nhid 200 --nlayers 2 --nhead 2
```

Adam 训练 GRU

```sh
python train_with_optim.py --data data/gigaspeech --cuda --epochs 20 --model GRU --lr 0.001 --emsize 200 --nhid 200 --no_tqdm
```
