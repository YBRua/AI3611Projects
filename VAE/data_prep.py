from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_mnist(root: str = './data/', ):
    """Fetch the MNIST dataset. Download if necessary.
    Dataset will be stored to `root` directory.

    Args:
        root (str): Directory to store the dataset. Defaults to './data/'.
    """
    train_dataset = datasets.MNIST(
        root=root, train=True,
        transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(
        root=root, train=False,
        transform=transforms.ToTensor(), download=False)

    return train_dataset, test_dataset


def wrap_dataloader(
        dataset: datasets.VisionDataset,
        batch_size: int = 64,
        shuffle: bool = True):
    """Wraps the given dataset into a torch `DataLoader`.

    Args:
        dataset (`VisionDataset`).
        batch_size (`int`, optional): Defaults to 64.
        shuffle (`bool`): Defaults to True.
    """
    return DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle)
