#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, tqdm_notebook

from datasets import PollyDataset
from model import ATModel
from utils import count_parameters

if __name__ == "__main__":

    polly = PollyDataset()
    batch_size = 1

    loader_train = torch.utils.data.DataLoader(
        polly,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    iter_train = iter(loader_train)
    print("Num batches:", len(loader_train))

    model = ATModel(
        in_channels=1,
        out_channels=3,
        kernel_size=3,
        stride=2,
        padding=1,
        batch_size=batch_size,
    )
    count_parameters(model)

    sample = iter_train.next()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)
    model.to(device)
    model.train()
    criterion = nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

    iter_train = iter(loader_train)

    steps = 10000
    img, labels = iter_train.next()  # retrieve minibatch
    img, labels = img.to(device), labels.to(device)

    epoch_pbar = tqdm_notebook(range(steps))

    for steps in epoch_pbar:
        output = model.forward(img)
        loss = criterion(output, labels.view(batch_size, -1))
        epoch_pbar.set_description("Loss: {}".format(str(loss.item())))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    epochs = 90
    model.train()

    writer = SummaryWriter()

    epoch_pbar = tqdm_notebook(range(epochs))

    for epoch in epoch_pbar:
        # training
        iter_train = iter(loader_train)
        offset = epoch * len(loader_train)  # training_iter offset
        data_pbar = tqdm_notebook(range(len(loader_train)))
        train_loss = 0
        bad_batches = 0
        for data in data_pbar:
            # hack to bypass dataloading error.
            # will result in lower loss than actual (since dividing by larger number)
            try:
                img, labels = iter_train.next()
            except Exception as e:
                print("Exception:", e)
                bad_batches += 1
            img, labels = img.to(device), labels.to(device)
            output = model.forward(img)

            loss = criterion(output, labels.view(batch_size, -1))
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            data_pbar.set_description("Training Loss: {}".format(str(loss.item())))
            global_batch_num = offset + data
            writer.add_scalar(
                "Loss/train", loss.item(), global_batch_num
            )  # plotting train loss over batch_num
            train_loss += loss.item()
        print("bad_batches:", bad_batches)
        train_loss /= len(loader_train)
        print("avg train loss:", train_loss)
        scheduler.step(train_loss)
        torch.cuda.empty_cache()
