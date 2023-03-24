import sys
import os
import numpy as np
# import json
from rich import print
import torch
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torchvision
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torchmetrics.classification import BinaryAccuracy

import exp_utils as exp_utils

from data_loader_even_pairwise import DataLoaderEvenPairwise

# https://github.com/guofei9987/pyLSHash
from pyLSHash import LSHash

# from superglue.models.matching import Matching

# define the LightningModule
class LSHModel(pl.LightningModule):
    def __init__(self, batch_size, freeze_backbone=True):
        super().__init__()
        self.save_hyperparameters() # save paramaters (matching_config) to checkpoint
        
        self.batch_size = batch_size

        self.freeze_backbone = freeze_backbone

        self.lsh = LSHash(hash_size=6, input_dim=8)

        # self.accuracy = BinaryAccuracy().to(self.device)

        self.test_datasets = None
        self.val_datasets = None
        
        self.model = nn.Sequential(
                # (1, 65+65, 50, 50)
                # nn.Conv2d(130, 64, kernel_size=3, stride=2, padding=1),
                # nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
                # nn.Flatten(),
                nn.Linear(1000, 32),
                # nn.LeakyReLU(),
                # nn.Linear(512, 256),
                # nn.LeakyReLU(),
                # nn.Linear(256, 64),
                # nn.LeakyReLU(),
                # nn.Linear(64, 1)
                )
        

    def backbone(self, sample):
        # print("sample", sample.shape) # shape (batch, 3, 400, 400)
        resnet18_model = torchvision.models.resnet18(pretrained=True).to(self.device)
        out = resnet18_model(sample) # shape (batch, 1000)
        return out

    def forward(self, sample):

        if self.freeze_backbone:
            with torch.no_grad():
                x = self.backbone(sample)
        else:
            x = self.backbone(sample)
        
        return x

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        name = dataloader_idx + 1
        if self.val_datasets is not None:
            name = self.val_datasets[dataloader_idx]
        
        self.evaluate(batch, name, "val")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        name = dataloader_idx + 1
        if self.test_datasets is not None:
            name = self.test_datasets[dataloader_idx]

        self.evaluate(batch, name, "test")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-5) # was 1e-3
        return optimizer
