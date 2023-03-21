import sys
import os
import numpy as np
from rich import print
import torch
from torch import optim, nn, utils, Tensor
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy
import exp_utils as exp_utils
from object_reid_sift import ObjectReIdSift
from shapely.geometry import Polygon, Point


# define the LightningModule
class SIFTModel(pl.LightningModule):
    def __init__(self, cutoff=0.5, visualise=False):
        super().__init__()
        self.save_hyperparameters() # save paramaters (matching_config) to checkpoint
        
        self.cutoff = cutoff
        self.visualise = visualise
        self.object_reid = ObjectReIdSift()

        self.accuracy = BinaryAccuracy().to(self.device)


    def evaluate(self, batch, dataloader_idx, stage=None):
        sample1, label1, list_poly1, sample2, label2, list_poly2 = batch
        
        ground_truth = (label1 == label2).float()
        ground_truth = torch.unsqueeze(ground_truth, 1)
        
        # iterate over the batch
        batch_result = []
        for i in np.arange(len(sample1)):
            img1 = sample1[i].detach().cpu().numpy()
            img1 = (img1 * 255).astype(dtype=np.uint8)
            img1 = np.squeeze(img1, axis=0)
            
            img2 = sample2[i].detach().cpu().numpy()
            img2 = (img2 * 255).astype(dtype=np.uint8)
            img2 = np.squeeze(img2, axis=0)

            poly1 = Polygon(np.array(list_poly1)[i])
            poly2 = Polygon(np.array(list_poly2)[i])
        
            result = self.object_reid.compare(img1, poly1, img2, poly2, visualise=self.visualise)
            if result is None:
                result = 0.0
            
            # print("result:", result)
            # print("gt:", ground_truth[i])

            result_bin = False
            if result > self.cutoff:
                result_bin = True
            
            # accuracy = ground_truth == result_bin

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
        self.log(f"{stage}_loss_{dataloader_idx + 1}", loss)

        acc = self.accuracy(batch_result, ground_truth)
        self.log(f"{stage}_acc_{dataloader_idx + 1}", acc, on_epoch=True)

    def validation_step(self, batch, batch_idx, dataloader_idx):
        self.evaluate(batch, dataloader_idx, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")
