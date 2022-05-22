import torch
import logging
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from datetime import datetime

from torchvision.models import resnet50
from torchvision.datasets import Places365
from torchvision import transforms

from torch.utils.data import DataLoader


def load_train_set():
    train_set = Places365(
        './finetune/places365/train',
        'train-standard',
        small=True,
        transform=transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ]),
        target_transform=None,
        download=True,
    )
    return train_set


def load_test_set():
    test_set = Places365(
        './finetune/places365/val',
        'val',
        small=True,
        transform=transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ]),
        target_transform=None,
        download=True,
    )
    return test_set


def setup_logger():
    logger = logging.getLogger('AVSC')
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter("%(message)s")

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    file_handler = logging.FileHandler(
        filename=f'./logs/{timestamp}-finetune.log',
        mode='a',
        encoding='utf-8')
    file_formatter = logging.Formatter(
        "[%(levelname)s %(asctime)s]: %(message)s")

    stream_handler.setFormatter(stream_formatter)
    file_handler.setFormatter(file_formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


model = resnet50(True)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
logger = setup_logger()

FINE_TUNE_EPOCHS = 50

train_set = load_train_set()
test_set = load_test_set()

train_loader = DataLoader(
    train_set,
    batch_size=32,
    shuffle=True)
test_loader = DataLoader(
    test_set,
    batch_size=32,
    shuffle=False)

best_eval_loss = float('inf')
for e in range(FINE_TUNE_EPOCHS):
    model.train()
    prog = tqdm(train_loader)
    tot_loss = 0
    tot_acc = 0
    for bid, (data, target) in enumerate(prog):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
        tot_acc += (output.argmax(1) == target).sum().item()
        avg_loss = tot_loss / (bid + 1)
        avg_acc = tot_acc / (bid + 1)
        prog.set_description(f'Epoch {e} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}')
    logger.info(f'Epoch: {e} | Loss: {avg_loss} | Acc: {avg_acc}')

    model.eval()
    test_prog = tqdm(test_loader)
    tot_loss = 0
    tot_acc = 0
    for bid, (data, target) in enumerate(test_prog):
        output = model(data)
        loss = loss_fn(output, target)
        tot_loss += loss.item() * data.size(0)
        tot_acc += (output.argmax(1) == target).sum().item()
    avg_loss = tot_loss / len(test_set)
    avg_acc = tot_acc / len(test_set)
    logger.info(f'Test | Loss: {avg_loss} | Acc: {avg_acc}')

    if avg_loss < best_eval_loss:
        best_eval_loss = avg_loss
        torch.save(model.state_dict(), './finetune/resnet50-places365.pt')
        logger.info('Model saved')
