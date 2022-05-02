# DevLog for AI3611 Project01 Language Modeling

## Experiment Roadmap

### Recurrent Neural Networks

#### Hidden Dimensions

- [ ] Default 200
- [ ] 256
- [ ] 512
- [ ] 768

#### Embedding Size

- [ ] Default 200
- [ ] 256
- [ ] 512
- [ ] 768

#### Architecture

- [ ] LSTM
- [ ] GRU
- [ ] RNN-RELU
- [ ] RNN-TANH

### Transformers

#### Layers

- [ ] Default 2
- [ ] 4
- [ ] 8
- [ ] 12

#### Heads

- [ ] Default 2
- [ ] 4
- [ ] 8

## Dump Results Here

### Prototyping

|      Model       | Epoches | M Params | Test PPL |                  Details                   |
| :--------------: | :-----: | :------: | :------: | :----------------------------------------: |
|       LSTM       |    6    |  11.75   |  164.97  |     [DevLog](#lstm-baseline-22-04-04)      |
|     LSTM-768     |    6    |  52.04   |  136.65  |                                            |
|       GRU        |    6    |  11.59   |  290.05  |      [DevLog](#gru-baseline-22-04-04)      |
|     GRU-Adam     |    6    |  11.59   |  154.61  |        [DevLog](#gru-adam-22-04-04)        |
|     GRU-768      |    6    |  49.68   |  171.29  |        [DevLog](#gru-768-22-04-05)         |
|   Transformer    |    6    |  11.60   |  762.20  |  [DevLog](#transformer-baseline-22-04-05)  |
| Transformer-Adam |    6    |  11.60   |  207.39  | [2Layer 2Head](#transformer-adam-22-04-05) |
| Transformer-Adam |    6    |  12.08   |  196.11  | [4Layer 2Head](#transformer-adam-22-04-05) |

### RNN ArchTune

|  Model   | Epoches | M Params | Test PPL | Details |
| :------: | :-----: | :------: | :------: | :-----: |
| RNN-RELU |    6    |  29.45   |  236.47  |

## 22-04-04 Init

### Experiments

#### LSTM Baseline 22-04-04

- `emsize: 200`
- `nhid: 200`
- `nlayers: 2`
- `lr: 20`

#### GRU Baseline 22-04-04

- `emsize: 200`
- `nhid: 200`
- `nlayers: 2`
- `lr: 20 -> 5`

#### GRU Adam 22-04-04

- `emsize: 200`
- `nhid: 200`
- `nlayers: 2`
- `lr: 0.001`

### 22-04-05 Transformers

#### GRU-768 22-04-05

- `emsize: 768`
- `nhid: 768`
- `nlayers: 2`
- `lr: 0.001`

#### Transformer Baseline 22-04-05

- `emsize: 200`
- `nhid: 200`
- `nlayers: 2`
- `nheads: 2`
- `lr: 20 -> 1.25`

#### Transformer Adam 22-04-05

- `emsize: 200`
- `nhid: 200`
- `nlayers: 2`
- `nheads: 2`
- `lr: 0.001`

#### Transformer-4 Adam 22-04-05

- `emsize: 200`
- `nhid: 200`
- `nlayers: 4`
- `nheads: 2`
- `lr: 0.001`

### 22-04-06 RNN Architecture Tuning

- For Architecture Tuning, we use embedding size `512` and hidden size `512`

#### ArchTune RNN-RELU 22-04-06
