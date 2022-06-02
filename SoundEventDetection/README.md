# AI3611 Sound Event Detection

## 快速开始

### 交我算超算平台

本项目代码位于 `/dssg/home/acct-stu/stu491/projects/SoundEventDetection` 目录下. 我们使用自行配置的环境进行实验. 我们在 `stu491` 目录下安装了 MiniConda，并在该 MiniConda 下配置了 `sed` 虚拟环境.

课程项目的代码同时保存在了 [Github仓库](https://github.com/YBRua/AI3611Projects)

#### 复现最优结果

> 最优结果为 Residual CRNN，带 0.2 Dropout数据增强为 Block Mixing，中值滤波大小为 25

- 要一键复现最优结果，请通过 `reproduce_best.slurm` 提交任务

```sh
sbatch reproduce_best.slurm
```

- 要复现其他结果，请参照已有的实验日志的配置 (`./experiments` 目录下) 调整配置文档
- 由于 GRU 的 Dropout 引入了一些随机性，实验结果 *不一定可以完全复现*，但是性能应该不会差太多

## Config YAML 的修改内容

为了支持数据增强以及一些新的自定义模型，我们对原始的 config YAML 的可选内容进行了一些扩充

- `augment`: 可选，如果有，则会使用数据增强
  - `mode`: 一个 List，指定了需要使用的数据增强类型. 数据增强会**按 List 中出现的顺序被作用于输入数据**
    - List 中的元素可以是 `time_shift` `spec_aug` `mixup` `xblock_mixing`
- `model`
  - `type`: 可以是 `Crnn` `GatedCrnn` 或 `ConvTransformer`，~~但是 `Crnn` 之外的模型都不太 work~~
  - `args`: 模型构造函数的 Keyword Arguments
    - `gru_layers`: GRU 的层数
    - `n_channels`: 3 个元素的 List. Crnn 默认有且只有 3 个卷积 Block，每个 Block 的 Output Channels 由这里指定
    - `pooling_sizes`: 3 个元素的 List. Pooling Layer 的 Kernel Size
    - `conv_block`: 卷积层的配置. 可以是 `conv` (默认的基线卷积层)，或 `res` (带残差连接的卷积层)
