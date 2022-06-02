# AI3611 Variational AutoEncoder

## 快速开始

### 交我算超算平台

本项目代码位于 `/dssg/home/acct-stu/stu491/projects/VAE` 目录下. 我们使用助教提供的 Conda 环境 (`espnet`) 进行实验.

课程项目的代码同时保存在了 [Github仓库](https://github.com/YBRua/AI3611Projects)

### 复现训练过程

> 最优 (测试集上的 Loss 最小) 模型配置为 MLP3-MLP3 结构的 VAE，Latent Space 大小 16

- 要一键复现最优结果，请通过 `reproduce_best.slurm` 提交任务

```sh
sbatch reproduce_best.slurm
```

### 复现采样/生成过程

采样和生成功能整合在 `train.py` 中，通过指定 `--skip_training` 参数可以跳过训练过程，直接加载预训练的模型并进行采样. 此时建议通过 `--model_save` 参数指定预训练模型的路径 (如果不指定，则会尝试从默认保存的路径加载一个模型，默认的模型名字为 `model[-AE]_<ENCODER>_<DECODER>_<LATENT_DIM>.pt`，保存在 `./ckpts/` 目录下)

一个仅复现测试、采样和生成过程的示例代码

```sh
python train.py --encoder MLP3 --decoder MLP3 --z_dim 2 --skip_training --model_save ./ckpts/model_MLP3_MLP3_2.pt
```

- 生成的图片保存在 `./imgaes/` 目录下
  - `image-recon.png` 是某一个 batch 重构后的图片
  - `image-x.png` 是某一个 batch 的原图片
  - `sample.png` 是从模型隐层空间随机采样后生成的图片
  - `loss-curve.png` 是训练时 Loss 的曲线
  - Latent Dim = 1 的模型会额外生成 `line-sample.png` ([-5, 5] 等间距采样后生成的图片)
  - Latent Dim = 2 的模型会额外生成 `grid-sample.png` (二维隐层空间等间距采样后生成的图片)
  - Latent Dim = 2 的模型会额外生成 `latent-space.pn` (测试集样本在二维隐层空间的散点图)

## 命令行参数

- `--data_root`: 数据集存放位置，默认为 `./data`
- `--batch_size`: 默认 64
- `--epochs`: 默认 50
- `--lr`: 默认 1e-3
- `--device`: 默认 `cuda`
- `--seed`: 随机数种子. 默认 42
- `--skip_training`: 是否跳过训练
- `--model_save`: 模型权重保存路径，如果为空，则默认保存到 `model[-AE]_<ENCODER>_<DECODER>_<LATENT_DIM>.pt`
- `--ae`: 若开启，则模型架构为 AE，否则默认为 VAE
- `--encoder`: 编码器架构. 可选为 `MLP` `MLP3` `CONV`. 默认 `MLP3`
- `--decoder`: 解码器架构. 可选为 `MLP` `MLP3` `CONV`. 默认 `MLP3`
- `--alpha`: 重构损失和 KLDiv 的权重，默认为 1
- `--z_dim`: 隐层空间大小. 默认为 2

## 文件说明

- `train.py`: 训练、测试的主要脚本
- `common.py`: 一些通用工具函数
- `data_prep.py`: 加载数据集
- `loss_funcs.py`: AE 和 VAE 的损失函数
- `vaeplotlib.py`: 负责采样和可视化作图
- `models`: 存放了 AE、VAE 的框架代码，以及各种 Encoder 和 Decoder
