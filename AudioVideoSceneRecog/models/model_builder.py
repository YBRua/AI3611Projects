import torch.nn as nn

from .late_baseline import LateBaseline
from .early_baseline import EarlyBaseline
from .mean_concat_dense import MeanConcatDense
from .unimodal_baseline import UnimodalBaseline

from .hybrid_baseline import HybridBaseline
from .late_weighted import LateWeighted
from .mma import BottleneckAttention
from .mma import MultiModalAttention

from typing import Dict


def get_model(config: Dict) -> nn.Module:
    model_arch = config["model"]
    if model_arch == "mean_concat_dense":
        model = MeanConcatDense(512, 512, config["num_classes"])
    elif model_arch == "early_baseline":
        model = EarlyBaseline(512, 512, config["num_classes"])
    elif model_arch == "late_baseline":
        model = LateBaseline(512, 512, config["num_classes"])
    elif model_arch == "unimodal_audio":
        model = UnimodalBaseline(512, 512, config["num_classes"], "audio")
    elif model_arch == "unimodal_video":
        model = UnimodalBaseline(512, 512, config["num_classes"], "video")
    elif model_arch == 'late_weighted':
        model = LateWeighted(512, 512, config["num_classes"])
    elif model_arch == 'hybrid_baseline':
        model = HybridBaseline(512, 512, config["num_classes"])
    elif model_arch == "multi_modal_attention":
        model = MultiModalAttention(512, 512, config["num_classes"])
    elif model_arch == "bottleneck_attention":
        model = BottleneckAttention(512, 512, config["num_classes"])
    else:
        raise ValueError(f"No model named {model_arch}")
    return model
