import torch
import torch.nn as nn


class EarlyBaseline(nn.Module):

    def __init__(self, audio_emb_dim, video_emb_dim, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ffwd = nn.Sequential(
            nn.Linear(audio_emb_dim + video_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256)
        )
        self.output = nn.Sequential(
            nn.Linear(256, 128),
            nn.Linear(128, self.num_classes),
        )

    def forward(
            self,
            audio_feat: torch.Tensor,
            video_feat: torch.Tensor):
        # audio_feat: [batch_size, time_steps, feat_dim]
        # video_feat: [batch_size, time_steps, feat_dim]
        early_cat = torch.cat([
            audio_feat, video_feat
        ], dim=-1)
        early_mean = early_cat.mean(1)
        out = self.ffwd(early_mean)
        out = self.output(out)
        return out
