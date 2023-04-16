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
class PairWiseClassifier3Model(pl.LightningModule):
    def __init__(self, batch_size, learning_rate, weight_decay, cutoff=0.9, freeze_backbone=True, visualise=True):
        super().__init__()
        self.save_hyperparameters() # save paramaters (matching_config) to checkpoint
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.cutoff = cutoff
        self.freeze_backbone = freeze_backbone
        self.visualise = visualise
        self.accuracy = BinaryAccuracy().to(self.device)
        # self.criterion = nn.BCEWithLogitsLoss() # This loss combines a Sigmoid layer and BCELoss in one class
        self.criterion = nn.CosineEmbeddingLoss()

        self.test_datasets = None
        self.val_datasets = None
        
        # self.backbone_model = torchvision.models.resnet18(pretrained=True).to(self.device)
        self.backbone_model = torchvision.models.resnet50(pretrained=True).to(self.device)

        # remove last layer: 
        self.backbone_model = torch.nn.Sequential(*(list(self.backbone_model.children())[:-1]))

        #   (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
        #   (fc): Linear(in_features=2048, out_features=1000, bias=True)

        if self.freeze_backbone:
            for param in self.backbone_model.parameters():
                param.requires_grad = False

        # TODO: make model better
        self.model = nn.Sequential(
                nn.Linear(2048, 256),
                nn.LeakyReLU(),
                # nn.Dropout(p=0.1),
                # nn.BatchNorm2d(100), #! parameter not right
                nn.Linear(256, 64)
                )
        
    def backbone(self, sample):
        # print("sample", sample.shape) # shape (batch, 3, 400, 400)
        out = self.backbone_model(sample) # shape (batch, 2048, 1, 1)
        out = torch.squeeze(out) # shape(batch, 2048)

        return out

    def forward(self, sample1, sample2):

        x1 = self.backbone(sample1)
        x2 = self.backbone(sample2)
        
        x1_out = self.model(x1)
        x2_out = self.model(x2)
        
        return x1_out, x2_out
    
    def run(self, label1, label2, sample1, sample2):
        # ground_truth = (label1 == label2)
        ground_truth = (label1 == label2).float()
        # ground_truth = torch.unsqueeze(ground_truth, 1)
        
        # run the forward step
        x1_out, x2_out = self(sample1, sample2)

        # print("x1_out.shape", x1_out.shape)
        
        loss = self.criterion(x1_out, x2_out, ground_truth)

        # compute accuracy
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        output = cos(x1_out, x2_out)
        output_thresh = torch.where(output > self.cutoff, 1.0, 0.0) # cutoff should be around 0.9

        # print("output_thresh.shape", output_thresh.shape)
        # print("ground_truth.shape", ground_truth.shape)

        acc = self.accuracy(output_thresh, ground_truth)

        return loss, acc

    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            # check if we are really frozen:
            for name, param in self.backbone_model.named_parameters():
                if param.requires_grad == True:
                    print("[red]backbone isn't frozen![/red]")
                    print(name, param.requires_grad)


        sample1, label1, dets1, sample2, label2, dets2 = batch
        
        loss, acc = self.run(label1, label2, sample1, sample2)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('train/acc', acc, on_step=True, on_epoch=True, batch_size=self.batch_size)

        if batch_idx == 99:
            print("train loss:", loss)
            print("train acc:", acc)

        return loss
    
    
    def evaluate(self, batch, name, stage=None):
        sample1, label1, dets1, sample2, label2, dets2 = batch
        
        loss, acc = self.run(label1, label2, sample1, sample2)

        print(f"{stage}/{name} acc:", acc)

        self.log(f"{stage}/{name}/loss_epoch", loss, on_step=False, on_epoch=True, batch_size=self.batch_size)

        self.log(f"{stage}/{name}/acc_epoch", acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)

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
