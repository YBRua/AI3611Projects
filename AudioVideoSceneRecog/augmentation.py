import torch
import numpy as np

from torch import Tensor


def mixup(
        audio_feats: Tensor,
        video_feats: Tensor,
        targets: Tensor,
        n_samples: int = 8,
        alpha: float = 0.1):
    lamb = np.random.beta(alpha, alpha)
    idx1 = torch.randperm(
        len(audio_feats), dtype=torch.long)[:n_samples]
    idx2 = torch.randperm(
        len(audio_feats), dtype=torch.long)[:n_samples]
    audio1 = audio_feats[idx1]
    audio2 = audio_feats[idx2]
    video1 = video_feats[idx1]
    video2 = video_feats[idx2]
    targets1 = targets[idx1]
    targets2 = targets[idx2]

    audio_aug = lamb * audio1 + (1 - lamb) * audio2
    video_aug = lamb * video1 + (1 - lamb) * video2
    targets_extra = lamb * targets1 + (1 - lamb) * targets2

    return audio_aug, video_aug, targets_extra


def uniform_noise(
        audio_feats: Tensor,
        video_feats: Tensor,
        targets: Tensor,
        bounds: float = 0.2):
    noise = torch.empty_like(audio_feats).uniform_(
        -bounds, bounds)
    audio_aug = audio_feats + noise
    noise = torch.empty_like(video_feats).uniform_(
        -bounds, bounds)
    video_aug = video_feats + noise

    return audio_aug, video_aug, targets
