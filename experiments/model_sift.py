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
    def __init__(self, batch_size, cutoff=0.5, visualise=False):
        super().__init__()
        self.save_hyperparameters() # save paramaters (matching_config) to checkpoint
        
        self.batch_size = batch_size
        self.cutoff = cutoff
        self.visualise = visualise
        self.object_reid = ObjectReIdSift()

        self.accuracy = BinaryAccuracy().to(self.device)
        self.loss_criterion = nn.BCEWithLogitsLoss() # This loss combines a Sigmoid layer and BCELoss in one class

        self.test_datasets = None
        self.val_datasets = None

        self.moving_avg_len = 20
        self.cutoffs = [cutoff] * self.moving_avg_len


    def run(self, sample1, label1, list_poly1, sample2, label2, list_poly2, train=True):
        ground_truth = (label1 == label2).float()
        # ground_truth = torch.unsqueeze(ground_truth, 1)
        
        # iterate over the batch
        results = []
        for i in np.arange(len(sample1)):
            img1 = exp_utils.torch_to_np_img(sample1[i])
            img2 = exp_utils.torch_to_np_img(sample2[i])

            poly1 = Polygon(np.array(list_poly1)[i])
            poly2 = Polygon(np.array(list_poly2)[i])
        
            result = self.object_reid.compare(img1, poly1, img2, poly2, visualise=self.visualise)

            results.append(result)

        results = np.array(results)


        # TODO: make this work 
        if train:
            pos_idxs = (ground_truth == 1).nonzero().detach().cpu().numpy()
            neg_idxs = (ground_truth == 0).nonzero().detach().cpu().numpy()

            print(f"pos_idxs {pos_idxs}")
            print(f"neg_idxs {neg_idxs}")

            print(f"result[pos_idxs] {results[pos_idxs]}")
            print(f"result[neg_idxs] {results[neg_idxs]}")

            # find cutoff parameter
            median_p = np.median(results[pos_idxs])
            median_n = np.median(results[neg_idxs])

            print(f"median_p {median_p}")
            print(f"median_n {median_n}")

            # we want median_n < median_p
            if median_n < median_p:
                new_cutoff = ((median_p - median_n)/2) + median_n
                self.cutoffs.append(new_cutoff)

                # moving average for updating cutoff value
                last_cut_offs = self.cutoffs[-self.moving_avg_len:]
                self.cutoff = np.mean(last_cut_offs)
                print(f"self.cutoff {self.cutoff}")
        

        results_bin = np.where(results > self.cutoff, 1.0, 0.0) 

        batch_result = torch.FloatTensor(results_bin).to(self.device)
        # batch_result = torch.unsqueeze(batch_result, 1).to(self.device)

        loss = self.loss_criterion(batch_result, ground_truth)
        acc = self.accuracy(batch_result, ground_truth)

        return loss, acc


    def training_step(self, batch, batch_idx):
        # TODO: learn cutoff parameter
        # use training only to learn cutoff parameter
        sample1, label1, list_poly1, sample2, label2, list_poly2 = batch

        loss, acc = self.run(sample1, label1, list_poly1, sample2, label2, list_poly2, train=True)

        self.log("train/cutoff", self.cutoff, on_step=True, on_epoch=True, batch_size=self.batch_size)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('train/acc', acc, on_step=True, on_epoch=True, batch_size=self.batch_size)

        # return loss
        return  torch.autograd.Variable(loss, requires_grad = True)


    def evaluate(self, batch, name, stage=None):
        sample1, label1, list_poly1, sample2, label2, list_poly2 = batch
        
        loss, acc = self.run(sample1, label1, list_poly1, sample2, label2, list_poly2, train=False)

        # if self.visualise:
        #     print("batch_result", batch_result.shape, type(batch_result))
        #     print("batch_result", batch_result)
        #     print("ground_truth", ground_truth.shape, type(ground_truth))
        #     print("ground_truth", ground_truth)

        self.log(f"{stage}/{name}/loss_epoch", loss, on_epoch=True, batch_size=self.batch_size, add_dataloader_idx=False)
        
        self.log(f"{stage}/{name}/acc_epoch", acc, on_epoch=True, prog_bar=True, batch_size=self.batch_size, add_dataloader_idx=False)

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

    #! only here to satisfy pytorch lightning
    def configure_optimizers(self):
        # optimizer = optim.Adam(self.parameters(), 
        #                        lr=1e-5,
        #                        )
        # return optimizer
        return None
    
    def on_save_checkpoint(self, checkpoint):
        # we update the cutoff parameter during training and want to save this new value
        checkpoint["learned_cutoff"] = self.cutoff

    def on_load_checkpoint(self, checkpoint):
        # load the learned cutoff parameter
        if "learned_cutoff" in checkpoint:
            self.cutoff = checkpoint["learned_cutoff"]
            self.cutoffs = [self.cutoff] * self.moving_avg_len