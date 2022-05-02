import torch
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models import VAE, AE
from typing import List, Union


matplotlib.style.use('seaborn-white')
sns.set_theme(context='paper', style='white', font_scale=1.5)


def gen_line_sample(
        low: float,
        high: float,
        step_size: float,
        model: VAE,
        device):
    with torch.no_grad():
        zs = torch.arange(low, high, step_size, dtype=torch.float32)
        zs = zs.reshape(-1, 1)
        zs = zs.to(device)
        samples = model.decode(zs)

    return samples


def gen_grid_sample(
        low: float,
        high: float,
        step_size: float,
        z_dim: int,
        model: VAE,
        device):
    with torch.no_grad():
        zs = torch.cartesian_prod(
            *[
                torch.arange(low, high, step_size, dtype=torch.float32)
                for _ in range(z_dim)
            ])
        zs = zs.reshape(-1, z_dim)
        zs = zs.to(device)
        samples = model.decode(zs)

    return samples


def plot_line_sample(
        low: float,
        high: float,
        step_size: float,
        model: VAE,
        device: torch.device,
        file_postfix: str):
    samples = gen_line_sample(low, high, step_size, model, device)
    samples = samples.reshape(-1, 1, samples.shape[-2], samples.shape[-1])
    save_image(
        samples, f'./images/line-sample{file_postfix}.png', nrow=20)


def plot_grid_sample(
        low: float,
        high: float,
        step_size: float,
        model: VAE,
        device: torch.device,
        file_postfix: str):
    """Generates and plots a 2D-grid of samples from the latent space.
    Figure will be saved to ./images/grid-sample{file_postfix}.png

    Args:
        low (float): Lower bound of latent space.
        high (float): Upper bound of latent space.
        step_size (float): Step size of grid.
        model (VAE): VAE model.
        device (_type_): Device to use.
        file_postfix (str): Postfix for the saved file.
            Used to distinguish different models
    """
    samples = gen_grid_sample(low, high, step_size, 2, model, device)
    samples = samples.reshape(-1, 1, samples.shape[-2], samples.shape[-1])
    save_image(
        samples, f'./images/grid-sample{file_postfix}.png',
        nrow=int(samples.shape[0] ** 0.5))


def plot_latent_space(
        dataloader: DataLoader,
        model: Union[VAE, AE],
        device: torch.device):
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = {k: ([], []) for k in range(10)}
    with torch.no_grad():
        for x, label in dataloader:
            x = x.to(device)
            z = model.encode(x).cpu().detach().numpy()  # B, Z
            zxs = z[:, 0].tolist()
            zys = z[:, 1].tolist()
            for zx, zy, lbl in zip(zxs, zys, label):
                labels[lbl.item()][0].append(zx)
                labels[lbl.item()][1].append(zy)

    for key, (xs, ys) in labels.items():
        ax.scatter(xs, ys, label=key, s=5)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), markerscale=4)
    ax.set_xlabel('z1')
    ax.set_ylabel('z2')
    ax.set_title('Scatter Plot of Latent Space')
    fig.tight_layout()

    return fig, ax


def loss_curve(loss: List, recon_loss: List, hidden_loss: List):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    ax: plt.Axes
    for axid, ax in enumerate(axes):
        data = [loss, recon_loss, hidden_loss][axid]
        y_label = ['Loss', 'Recon Loss', 'KLDiv Loss'][axid]

        ax.plot(data)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(y_label)
        ax.set_title(f'Test {y_label}')

    fig.tight_layout()

    return fig, axes
