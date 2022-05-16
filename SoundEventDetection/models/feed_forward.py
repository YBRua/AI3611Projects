import torch
import torch.nn as nn


class LinearFeedFwd(nn.Module):
    def __init__(self, input_dims: int, output_dims: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dims, output_dims)

    def forward(self, x: torch.Tensor):
        return self.fc(x)


class LocalizationFeedFwd(nn.Module):
    def __init__(self, input_dims: int, output_dims: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dims, output_dims)
        self.masker = nn.Linear(output_dims, output_dims)

    def forward(self, x: torch.Tensor):
        out = torch.sigmoid(self.fc(x))
        mask = torch.softmax(self.masker(out), dim=-1)
        return torch.multiply(out, mask), mask
