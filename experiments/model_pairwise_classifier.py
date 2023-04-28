import sys
import os
import numpy as np
# import json
from rich import print
import torch
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torchmetrics.classification import BinaryAccuracy

import exp_utils as exp_utils

from data_loader_even_pairwise import DataLoaderEvenPairwise

from superglue.models.matching import Matching

# define the LightningModule
class PairWiseClassifierModel(pl.LightningModule):
    def __init__(self, batch_size, learning_rate, weight_decay, opt=None, freeze_backbone=True):
        super().__init__()
        self.save_hyperparameters() # save paramaters (matching_config) to checkpoint
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        matching_config = {
            'superpoint': {
                'nms_radius': opt.nms_radius,
                'keypoint_threshold': opt.keypoint_threshold,
                'max_keypoints': opt.max_keypoints
            },
            'superglue': {
                'weights': opt.superglue,
                'sinkhorn_iterations': opt.sinkhorn_iterations,
                'match_threshold': opt.match_threshold,
            }
        }

        self.freeze_backbone = freeze_backbone
        self.matching = Matching(matching_config).to(self.device)
        self.superpoint= self.matching.superpoint

        self.accuracy = BinaryAccuracy().to(self.device)

        self.test_datasets = None
        self.val_datasets = None

        # TODO: make model better
        self.model = nn.Sequential(
                # (1, 65+65, 50, 50)
                nn.Conv2d(130, 64, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
                nn.Flatten(),
                nn.Linear(1875, 1024),
                nn.LeakyReLU(),
                nn.Linear(1024, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 1)
                )
        
        # https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/cifar10-baseline.html
        # ? maybe I should end with F.log_softmax(out, dim=1)
        # ? then do: loss = F.nll_loss(logits, y)

    def backbone(self, sample):
        
        #! I don't think it has a .parameters()
        print("[red]THIS PROBABLY DOESN'T WORK!!![/red]")
        if self.freeze_backbone:
            for param in self.superpoint.parameters():
                param.requires_grad = False

        img_data = self.superpoint({'image': sample})
        x = img_data["x_waypoint"] # shape: (1, 65, 50, 50)
        return x

    def forward(self, sample1, sample2):

        # self.superpoint.eval()
        # print("sample2.requires_grad", sample2.requires_grad)
        # if self.freeze_backbone:
        #     with torch.no_grad():
        #         x1 = self.backbone(sample1)
        #         x2 = self.backbone(sample2)
        # else:
        x1 = self.backbone(sample1)
        x2 = self.backbone(sample2)
        
        # print("x1.requires_grad", x1.requires_grad)
        
        x = torch.cat((x1, x2), 1) # shape: (1, 130, 50, 50)
        
        # print("x.requires_grad", x.requires_grad)
        # print("x.shape", x.shape)
        
        x_out = self.model(x) #! CHECK IF THIS HAS GRAD
        
        # print("x_out.requires_grad", x_out.requires_grad)
        
        return x_out
    
    def run(self, label1, label2, sample1, sample2):
        # ground_truth = (label1 == label2)
        ground_truth = (label1 == label2).float()
        ground_truth = torch.unsqueeze(ground_truth, 1)
        
        # run the forward step
        x_out = self(sample1, sample2)
        
        criterion = nn.BCEWithLogitsLoss() # This loss combines a Sigmoid layer and BCELoss in one class
        loss = criterion(x_out, ground_truth)
        acc = self.accuracy(x_out, ground_truth)

        return loss, acc

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
    
        sample1, label1, dets1, sample2, label2, dets2 = batch
        
        loss, acc = self.run(label1, label2, sample1, sample2)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('train/acc', acc, on_step=True, on_epoch=True, batch_size=self.batch_size)

        if batch_idx == 99:
            print("loss:", loss)
            print("acc:", acc)

        return loss

    def evaluate(self, batch, name, stage=None):
        sample1, label1, dets1, sample2, label2, dets2 = batch
        
        loss, acc = self.run(label1, label2, sample1, sample2)

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
        optimizer = optim.Adam(self.parameters(), 
                               lr=self.learning_rate, 
                               weight_decay=self.weight_decay)
        return optimizer
