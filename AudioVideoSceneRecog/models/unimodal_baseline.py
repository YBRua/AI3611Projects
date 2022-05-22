import torch
import torch.nn as nn


class UnimodalBaseline(nn.Module):

    def __init__(
            self,
            audio_emb_dim: int,
            video_emb_dim: int,
            num_classes: int,
            modality: str) -> None:
        super().__init__()
        if modality not in ['audio', 'video']:
            raise ValueError("modality must be either 'audio' or 'video'")

        self.num_classes = num_classes
        embd_dim = audio_emb_dim if modality == 'audio' else video_emb_dim
        self.modality = modality
        self.ffwd = nn.Sequential(
            nn.Linear(embd_dim, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(768, 256)
        )
        self.output = nn.Sequential(
            nn.Linear(256, 128),
            nn.Linear(128, self.num_classes),
        )

    def forward(
            self,
            audio_feat: torch.Tensor,
            video_feat: torch.Tensor):
        features = audio_feat if self.modality == 'audio' else video_feat
        feat_mean = features.mean(1)
        out = self.ffwd(feat_mean)
        out = self.output(out)
        return out
