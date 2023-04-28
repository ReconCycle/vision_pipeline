import sys
import os
import numpy as np
from rich import print
import torch
import cv2
from torch import optim, nn
import torchvision
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy
import clip
import exp_utils as exp_utils
import albumentations as A


class ClipModel(pl.LightningModule):
    def __init__(self, batch_size, opt=None, cutoff=0.9, visualise=False):
        super().__init__()
        self.save_hyperparameters() # save paramaters (matching_config) to checkpoint
        
        self.batch_size = batch_size
        self.cutoff = cutoff
        self.visualise = visualise

        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        self.accuracy = BinaryAccuracy().to(self.device)

        self.test_datasets = None
        self.val_datasets = None

    def backbone(self, sample):
        return self.model.encode_image(sample.to(self.device))


    def evaluate(self, batch, name, stage=None):
        sample1, label1, dets1, sample2, label2, dets2 = batch
        
        ground_truth = (label1 == label2).float()
        # ground_truth = torch.unsqueeze(ground_truth, 1)

        image1_features = self.backbone(sample1)
        image2_features = self.backbone(sample2)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        output = cos(image1_features, image2_features)

        if self.visualise:
            print(f"sample1.shape {sample1.shape}")
            print(f"image1_features.shape {image1_features.shape}")
            print(f"output.shape {output.shape}")
            print(f"output: {output}")
            print(f"gt: {ground_truth}")

        output_thresh = torch.where(output > self.cutoff, 1.0, 0.0) # cutoff should be around 0.9

        # compute accuracy
        acc = self.accuracy(output_thresh, ground_truth)
        self.log(f"{stage}/{name}/acc", acc, on_epoch=True, prog_bar=True,batch_size=self.batch_size, add_dataloader_idx=False)

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
