import torch
import torch.nn.functional as F

from torch import Tensor


def manual_kl_divergence(mu: Tensor, log_sigma: Tensor) -> Tensor:
    sigma = torch.exp(log_sigma)
    return torch.sum(
        0.5 * (torch.pow(mu, 2) + torch.pow(sigma, 2) - 2 * log_sigma - 1),
        dim=1
    )


class BCELoss():
    def __init__(self) -> None:
        pass

    def __call__(
        self,
        x: Tensor,
        x_recon: Tensor
    ) -> Tensor:
        assert x.shape[0] == x_recon.shape[0]

        B = x.shape[0]
        x = x.reshape(x.shape[0], -1)
        x_recon = x_recon.reshape(x_recon.shape[0], -1)

        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
        return recon_loss / B


class BCEKLDLoss():
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

        self.recon_loss = F.binary_cross_entropy
        # self.hidden_loss = kl_divergence
        self.hidden_loss = manual_kl_divergence

    def __call__(
            self,
            x: Tensor,  # B, H, W
            x_recon: Tensor,  # B, H, W
            mu: Tensor,  # B, Z
            log_sigma: Tensor):  # B, Z

        B = x.shape[0]
        x = x.reshape(x.shape[0], -1)  # B, H * W
        x_recon = x_recon.reshape(x_recon.shape[0], -1)  # B, H * W

        # reconstruction loss
        recon_loss = self.recon_loss(x_recon, x, reduction='sum') / B

        # KL-Div loss
        hidden_loss = self.hidden_loss(mu, log_sigma)
        hidden_loss = hidden_loss.sum() / B

        return (recon_loss + self.alpha * hidden_loss), recon_loss, hidden_loss
