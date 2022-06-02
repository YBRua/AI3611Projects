# AI3611 Sound Event Detection

## 快速开始

### 交我算超算平台

本项目代码位于 `/dssg/home/acct-stu/stu491/projects/AudioVideoSceneRecog` 目录下. 我们使用自行配置的环境进行实验. 我们在 `stu491` 目录下安装了 MiniConda，并在该 MiniConda 下配置了 `sed` 虚拟环境.

课程项目的代码同时保存在了 [Github仓库](https://github.com/YBRua/AI3611Projects)

#### 复现最优结果

> 最优结果为 Late-Weighted 模型

- 要一键复现最优结果，请通过 `reproduce_best.slurm` 提交任务

```sh
sbatch reproduce_best.slurm
```

#### 复现其他结果

- 要复现其他结果，请参照已有的实验日志的配置 (`./configs` 目录下) 调整配置

由于完整的训练-测试-评价过程需要运行 3 个不同的 Python 文件，为了方便起见，我们整合了 `just_rush.sh` 用于快速运行整个流水线.

```sh
source just_rush.sh <config_file_name>
```

其中 `config_file_name>` 是配置文件的名称，例如 `baseline` 或者 `late_weighted` 等 (参见 `./configs` 目录下的 YAML 配置文件)

## 文件改动

- `models` 目录下存放了所有可用的模型架构的代码
- `fine_tune.py` 是尝试参照 SOTA 在 Places365 上 Fine Tune ResNet50，但是因为耗时太长了没有实际跑完
- `augmentation.py` 是一些数据增强方法
- `plotter.py` 是写报告时使用的可视化代码，训练-测试过程不需要用到这里的代码
