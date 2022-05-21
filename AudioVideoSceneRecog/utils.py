import sys
import torch.nn as nn
from loguru import logger


def getfile_outlogger(outputfile):
    log_format = "[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])
    if outputfile:
        logger.add(outputfile, enqueue=True, format=log_format)
    return logger


def _format_params(n_params: int):
    if n_params < 1e3:
        return n_params
    elif n_params < 1e6:
        return f"{n_params / 1e3:.2f}K"
    elif n_params < 1e9:
        return f"{n_params / 1e6:.2f}M"


def count_parameters(model: nn.Module):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return _format_params(n_params)
