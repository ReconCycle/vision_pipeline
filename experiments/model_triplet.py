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

# inspiration
# https://www.kaggle.com/code/hirotaka0122/triplet-loss-with-pytorch/notebook
# https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html

# define the LightningModule
class TripletModel(pl.LightningModule):
    def __init__(self, batch_size, freeze_backbone=True):
        super().__init__()
        self.save_hyperparameters() # save paramaters (matching_config) to checkpoint
        
        self.batch_size = batch_size
        self.freeze_backbone = freeze_backbone
        self.acc_criterion = BinaryAccuracy().to(self.device)
        self.test_datasets = None
        self.val_datasets = None

        self.resnet18_model = torchvision.models.resnet18(pretrained=True).to(self.device)
        # TODO: make model better
        self.model = nn.Sequential(
                nn.Linear(1000, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 64),
                # nn.LeakyReLU(),
                # nn.Linear(64, 1)
                )
        

    def backbone(self, sample):
        # print("sample", sample.shape) # shape (batch, 3, 400, 400)
        out = self.resnet18_model(sample) # shape (batch, 1000)
        out = self.model(out)
        return out

    def forward(self, sample):
        
        #! doesn't make sense to freeze when we are only using backbone with no additional layers
        # if self.freeze_backbone:
        #     with torch.no_grad():
        #         x = self.backbone(sample)
        # else:
        x = self.backbone(sample)
        
        # print("x1.requires_grad", x1.requires_grad)
        
        # print("x.requires_grad", x.requires_grad)
        # print("x.shape", x.shape)
        
        # x_out = self.model(x) #! CHECK IF THIS HAS GRAD
        
        # print("x_out.requires_grad", x_out.requires_grad)
        
        return x

    def accuracy(self, a_out, p_out, n_out, a_label, p_label, n_label):
        # determine accuracy
        dist_criterion = nn.PairwiseDistance(p=2)
        dist_p = dist_criterion(a_out, p_out)
        dist_n = dist_criterion(a_out, n_out)

        # ? we could measure acc like this instead:
        # ((dist_p - dist_n) > 0).sum()

        dist = torch.vstack((torch.unsqueeze(dist_p, 1), 
                             torch.unsqueeze(dist_n, 1)))

        cutoff = 1.0 # value below cutoff is positive, above is negative
        result = dist < cutoff

        ground_truth_n = (a_label == n_label).float()
        ground_truth_n = torch.unsqueeze(ground_truth_n, 1)

        ground_truth_p = (a_label == p_label).float()
        ground_truth_p = torch.unsqueeze(ground_truth_p, 1)

        ground_truth = torch.vstack((ground_truth_p, ground_truth_n))

        acc = self.acc_criterion(result, ground_truth)

        return acc

    def training_step(self, batch, batch_idx):
        a_sample, a_label, a_dets, \
            p_sample, p_label, p_dets, \
                n_sample, n_label, n_dets = batch
        
        a_out = self(a_sample)
        p_out = self(p_sample)
        n_out = self(n_sample)

        criterion = nn.TripletMarginLoss(margin=1.0, p=2)

        loss = criterion(a_out, p_out, n_out)

        acc = self.accuracy(a_out, p_out, n_out, a_label, p_label, n_label)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('train/acc', acc, on_step=True, on_epoch=True, batch_size=self.batch_size)

        # if batch_idx == 99:
        #     print("x_out", x_out)
        #     print("ground truth", ground_truth)
        #     print("loss:", loss)
        #     print("acc:", acc)

        return loss

    def evaluate(self, batch, name, stage=None):
        a_sample, a_label, a_dets, \
            p_sample, p_label, p_dets, \
                n_sample, n_label, n_dets = batch
        
        a_out = self(a_sample)
        p_out = self(p_sample)
        n_out = self(n_sample)

        criterion = nn.TripletMarginLoss(margin=1.0, p=2)

        loss = criterion(a_out, p_out, n_out)

        acc = self.accuracy(a_out, p_out, n_out, a_label, p_label, n_label)

        # TODO come up with a cutoff for when the distance is small enough to be called the same
        # for i in np.arange(len(ground_truth)):
        #     if ground_truth[i]:
        #         print(f"positive: {dist[i]}")
        #     else:
        #         print(f"negative: {dist[i]}")

        self.log(f"{stage}/{name}/loss_epoch", loss, on_step=False, on_epoch=True, batch_size=self.batch_size, add_dataloader_idx=False)

        self.log(f"{stage}/{name}/acc_epoch", acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size, add_dataloader_idx=False)

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
