import torch
import torch.nn as nn


class LateBaseline(nn.Module):
    def __init__(self, audio_emb_dim, video_emb_dim, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.audio_predictor = nn.Sequential(
            nn.Linear(audio_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, self.num_classes)
        )
        self.video_predictor = nn.Sequential(
            nn.Linear(video_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, self.num_classes)
        )

    def forward(
            self,
            audio_feat: torch.Tensor,
            video_feat: torch.Tensor):
        # audio_feat: [batch_size, time_steps, feat_dim]
        # video_feat: [batch_size, time_steps, feat_dim]
        audio_emb = audio_feat.mean(1)
        audio_pred = self.audio_predictor(audio_emb)

        video_emb = video_feat.mean(1)
        video_pred = self.video_predictor(video_emb)

        output = (audio_pred + video_pred) / 2

        return output
