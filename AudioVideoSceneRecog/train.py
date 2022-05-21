from pathlib import Path
import os
import argparse
import time
import random

from tqdm import tqdm
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from dataset import SceneDataset
import models
import utils


# commandline arguments
parser = argparse.ArgumentParser(description='training networks')
parser.add_argument(
    '--config_file', type=str, required=True)
parser.add_argument(
    '--seed', type=int, default=0, required=False,
    help='set the seed to reproduce result')
args = parser.parse_args()

# fix random seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# load config
with open(args.config_file, "r") as reader:
    config = yaml.load(reader, Loader=yaml.FullLoader)

# normalization
mean_std_audio = np.load(config["data"]["audio_norm"])
mean_audio = mean_std_audio["global_mean"]
std_audio = mean_std_audio["global_std"]
mean_std_video = np.load(config["data"]["video_norm"])
mean_video = mean_std_video["global_mean"]
std_video = mean_std_video["global_std"]


def audio_transform(x):
    return (x - mean_audio) / std_audio


def video_transform(x):
    return (x - mean_video) / std_video


# load data
tr_ds = SceneDataset(
    config["data"]["train"]["audio_feature"],
    config["data"]["train"]["video_feature"],
    audio_transform,
    video_transform)
tr_dataloader = DataLoader(
    tr_ds, shuffle=True, **config["data"]["dataloader_args"])
cv_ds = SceneDataset(
    config["data"]["cv"]["audio_feature"],
    config["data"]["cv"]["video_feature"],
    audio_transform,
    video_transform)
cv_dataloader = DataLoader(
    cv_ds, shuffle=False, **config["data"]["dataloader_args"])

# init logging
output_dir = config["output_dir"]
Path(output_dir).mkdir(exist_ok=True, parents=True)
logging_writer = utils.getfile_outlogger(os.path.join(output_dir, "train.log"))

# init model
model = models.MeanConcatDense(512, 512, config["num_classes"])
print(model)
loss_fn = torch.nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = getattr(optim, config["optimizer"]["type"])(
    model.parameters(),
    **config["optimizer"]["args"])
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    **config["lr_scheduler"]["args"])

print('-----------start training-----------')


def train(epoch):
    model.train()
    train_loss = 0.
    start_time = time.time()
    count = len(tr_dataloader) * (epoch - 1)
    loader = tqdm(tr_dataloader)
    for batch_idx, batch in enumerate(loader):
        count = count + 1
        audio_feat = batch["audio_feat"].to(device)
        video_feat = batch["video_feat"].to(device)
        target = batch["target"].to(device)

        # training
        optimizer.zero_grad()

        logit = model(audio_feat, video_feat)
        loss = loss_fn(logit, target)
        loss.backward()

        train_loss += loss.item()
        optimizer.step()

        if (batch_idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(
                f'| epoch {epoch:3d} ',
                f'| batch {batch_idx:5d}/{len(tr_dataloader):5d} ',
                f'| {elapsed * 1000 / (batch_idx + 1):5.2f} ms/batch ',
                f'| loss {loss.item:5.2f} |')

    train_loss /= (batch_idx + 1)
    logging_writer.info('-' * 99)
    logging_writer.info(
        f'| epoch {epoch:3d} '
        f'| time {time.time() - start_time:5.2f}s '
        f'| training loss {train_loss:5.2f} |')
    return train_loss


def validate(epoch):
    model.eval()
    start_time = time.time()
    # data loading
    cv_loss = 0.
    targets = []
    preds = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(cv_dataloader):
            audio_feat = batch["audio_feat"].to(device)
            video_feat = batch["video_feat"].to(device)
            target = batch["target"].to(device)
            logit = model(audio_feat, video_feat)
            loss = loss_fn(logit, target)
            pred = torch.argmax(logit, 1)
            targets.append(target.cpu().numpy())
            preds.append(pred.cpu().numpy())
            cv_loss += loss.item()

    cv_loss /= (batch_idx+1)
    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    accuracy = accuracy_score(targets, preds)
    logging_writer.info(
        f'| epoch {epoch:3d} '
        f'| time: {time.time() - start_time:5.2f}s '
        f'| val loss: {cv_loss:5.2f} '
        f'| val acc: {accuracy:5.2f} |')
    logging_writer.info('-' * 99)

    return cv_loss


training_loss = []
cv_loss = []


with open(os.path.join(output_dir, 'config.yaml'), "w") as writer:
    yaml.dump(config, writer, default_flow_style=False)

not_improve_cnt = 0
for epoch in range(1, config["epoch"]):
    print('epoch', epoch)
    training_loss.append(train(epoch))
    cv_loss.append(validate(epoch))
    print('-' * 99)
    print(
        f'Epoch {epoch}',
        f'Training Loss: {training_loss[-1]:.4f}'
        f'CV Loss: {cv_loss[-1]:.4f}')

    if cv_loss[-1] == np.min(cv_loss):
        # save current best model
        torch.save(
            model.state_dict(),
            os.path.join(output_dir, 'best_model.pt'))
        print('best validation model found and saved.')
        print('-' * 99)
        not_improve_cnt = 0
    else:
        not_improve_cnt += 1

    lr_scheduler.step(cv_loss[-1])

    if not_improve_cnt == config["early_stop"]:
        break


minmum_cv_index = np.argmin(cv_loss)
minmum_loss = np.min(cv_loss)
plt.plot(training_loss, 'r')
plt.plot(cv_loss, 'b')
plt.axvline(x=minmum_cv_index, color='k', linestyle='--')
plt.plot(minmum_cv_index, minmum_loss, 'r*')

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'val_loss'], loc='upper left')
plt.savefig(os.path.join(output_dir, 'loss.png'))
