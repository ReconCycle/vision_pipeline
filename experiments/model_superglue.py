import sys
import os
import numpy as np
from rich import print
import torch
import cv2
from torch import optim, nn
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy

import exp_utils as exp_utils
from object_reid_superglue import ObjectReIdSuperGlue

# define the LightningModule
class SuperGlueModel(pl.LightningModule):
    def __init__(self, batch_size, opt=None, cutoff=0.5, visualise=False):
        super().__init__()
        self.save_hyperparameters() # save paramaters (matching_config) to checkpoint
        
        self.batch_size = batch_size
        self.cutoff = cutoff
        self.visualise = visualise
        self.object_reid = ObjectReIdSuperGlue(opt=opt)

        self.accuracy = BinaryAccuracy().to(self.device)

        self.test_datasets = None
        self.val_datasets = None

    def evaluate(self, batch, name, stage=None):
        sample1, label1, dets1, sample2, label2, dets2 = batch
        
        ground_truth = (label1 == label2).float()
        ground_truth = torch.unsqueeze(ground_truth, 1)

        # iterate over batch
        batch_result = []
        for i in np.arange(len(sample1)):
            img1 = exp_utils.torch_to_np_img(sample1[i]).astype(np.float32)
            img2 = exp_utils.torch_to_np_img(sample2[i]).astype(np.float32)
        
            result = self.object_reid.compare(img1, img2, visualise=self.visualise)
            if result is None:
                result = 0.0
                
            result_bin = False
            if result > self.cutoff:
                result_bin = True
            

            batch_result.append(result_bin)

        batch_result = torch.FloatTensor(batch_result)
        batch_result = torch.unsqueeze(batch_result, 1).to(self.device)

        if self.visualise:
            print("batch_result", batch_result.shape, type(batch_result))
            print("batch_result", batch_result)
            print("ground_truth", ground_truth.shape, type(ground_truth))
            print("ground_truth", ground_truth)


        criterion = nn.BCEWithLogitsLoss() # This loss combines a Sigmoid layer and BCELoss in one class
        loss = criterion(batch_result, ground_truth)
        self.log(f"{stage}_{name}/loss", loss, batch_size=self.batch_size, add_dataloader_idx=False)

        acc = self.accuracy(batch_result, ground_truth)
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
