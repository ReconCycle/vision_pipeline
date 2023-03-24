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

# from superglue.models.matching import Matching

# define the LightningModule
class PairWiseClassifier2Model(pl.LightningModule):
    def __init__(self, batch_size, freeze_backbone=True):
        super().__init__()
        self.save_hyperparameters() # save paramaters (matching_config) to checkpoint
        
        self.batch_size = batch_size
        self.freeze_backbone = freeze_backbone
        self.accuracy = BinaryAccuracy().to(self.device)

        self.test_datasets = None
        self.val_datasets = None
        
        self.resnet18_model = torchvision.models.resnet18(pretrained=True).to(self.device)

        # TODO: make model better
        self.model = nn.Sequential(
                nn.Linear(2000, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 1)
                )
        

    def backbone(self, sample):
        # print("sample", sample.shape) # shape (batch, 3, 400, 400)
        
        out = self.resnet18_model(sample) # shape (batch, 1000)
        return out

    def forward(self, sample1, sample2):

        if self.freeze_backbone:
            with torch.no_grad():
                x1 = self.backbone(sample1)
                x2 = self.backbone(sample2)
        else:
            x1 = self.backbone(sample1)
            x2 = self.backbone(sample2)
        
        # print("x1.requires_grad", x1.requires_grad)
        
        x = torch.cat((x1, x2), 1) # shape: (1, 130, 50, 50)
        
        # print("x.requires_grad", x.requires_grad)
        # print("x.shape", x.shape)
        
        x_out = self.model(x) #! CHECK IF THIS HAS GRAD
        
        # print("x_out.requires_grad", x_out.requires_grad)
        
        return x_out

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
    
        sample1, label1, dets1, sample2, label2, dets2 = batch
        
        # ground_truth = (label1 == label2)
        ground_truth = (label1 == label2).float()
        ground_truth = torch.unsqueeze(ground_truth, 1)
        
        # run the forward step
        x_out = self(sample1, sample2)
        
        criterion = nn.BCEWithLogitsLoss() # This loss combines a Sigmoid layer and BCELoss in one class
        loss = criterion(x_out, ground_truth)
        acc = self.accuracy(x_out, ground_truth)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('train/acc', acc, on_step=True, on_epoch=True, batch_size=self.batch_size)


        if batch_idx == 99:
            print("x_out", x_out)
            print("ground truth", ground_truth)
            print("loss:", loss)
            print("acc:", acc)

        return loss

    def evaluate(self, batch, name, stage=None):
        sample1, label1, dets1, sample2, label2, dets2 = batch
        
        ground_truth = (label1 == label2).float()
        ground_truth = torch.unsqueeze(ground_truth, 1)
        
        # run the forward step
        x_out = self(sample1, sample2)
        
        criterion = nn.BCEWithLogitsLoss() # This loss combines a Sigmoid layer and BCELoss in one class
        loss = criterion(x_out, ground_truth)
        self.log(f"{stage}_{name}/loss_epoch", loss, on_step=False, on_epoch=True, batch_size=self.batch_size)

        acc = self.accuracy(x_out, ground_truth)
        self.log(f"{stage}_{name}/acc_epoch", acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)

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
