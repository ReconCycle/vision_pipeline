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
    def __init__(self, batch_size, learning_rate, weight_decay, cutoff=1.0, freeze_backbone=True, visualise=False):
        super().__init__()
        self.save_hyperparameters() # save paramaters (matching_config) to checkpoint
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.freeze_backbone = freeze_backbone
        self.cutoff = cutoff
        self.visualise = visualise
        self.acc_criterion = BinaryAccuracy().to(self.device)
        self.test_datasets = None
        self.val_datasets = None

        self.moving_avg_len = 20
        self.cutoffs = [cutoff] * self.moving_avg_len

        print(f"self.cutoffs {self.cutoffs}")

        # self.backbone_model = torchvision.models.resnet18(pretrained=True).to(self.device)
        self.backbone_model = torchvision.models.resnet50(pretrained=True).to(self.device)

        self.backbone_model = torch.nn.Sequential(*(list(self.backbone_model.children())[:-1]))
        
        # todo: remove last two layers:
        #   (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
        #   (fc): Linear(in_features=2048, out_features=1000, bias=True)

        # we freeze the backbone like this:
        # https://stackoverflow.com/questions/63785319/pytorch-torch-no-grad-versus-requires-grad-false
        if self.freeze_backbone:
            print("[red]Freezing backbone[/red]")
            for param in self.backbone_model.parameters():
                param.requires_grad = False

        # TODO: make model better
        self.model = nn.Sequential(
                nn.Linear(2048, 128),
                # nn.Linear(2048, 512),
                # nn.LeakyReLU(),
                # nn.Dropout(p=0.2),
                # # nn.BatchNorm2d(100), #! parameter not right
                # nn.Linear(512, 256),
                # nn.LeakyReLU(),
                # nn.Dropout(p=0.2),
                # # nn.BatchNorm2d(100), #! parameter not right
                # nn.Linear(256, 128),
                # nn.LeakyReLU(),
                # nn.Linear(64, 8),
                )
        
        # we could try: GlobalAveragePooling2D, Dropout, and BatchNormalization
        # https://pyimagesearch.com/2023/03/06/triplet-loss-with-keras-and-tensorflow/
        # x = layers.GlobalAveragePooling2D()(extractedFeatures)
        # x = layers.Dense(units=1024, activation="relu")(x)
        # x = layers.Dropout(0.2)(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.Dense(units=512, activation="relu")(x)
        # x = layers.Dropout(0.2)(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.Dense(units=256, activation="relu")(x)
        # x = layers.Dropout(0.2)(x)
        # outputs = layers.Dense(units=128)(x)
        
        self.triplet_criterion = nn.TripletMarginLoss(margin=1.0, p=2)

    def backbone(self, sample):
        # print("sample", sample.shape) # shape (batch, 3, 400, 400)
        out = self.backbone_model(sample) # shape (batch, 1000)
        out = torch.squeeze(out) # shape(batch, 2048)

        return out

    def forward(self, sample):
        
        x = self.backbone(sample)

        # print("x.requires_grad", x.requires_grad)

        x = self.model(x)

        # print("x.requires_grad", x.requires_grad)
        
        # print("x.requires_grad", x.requires_grad)
        # print("x.shape", x.shape)
        
        # x_out = self.model(x) #! CHECK IF THIS HAS GRAD
        
        # print("x_out.requires_grad", x_out.requires_grad)
        
        return x

    def accuracy(self, a_out, p_out, n_out, a_label, p_label, n_label, visualise=False, train=True):
        # determine accuracy
        dist_criterion = nn.PairwiseDistance(p=2)
        dist_p = dist_criterion(a_out, p_out)
        dist_n = dist_criterion(a_out, n_out)

        # ? we could measure acc like this instead:
        # ((dist_p - dist_n) > 0).sum()

        dist = torch.vstack((torch.unsqueeze(dist_p, 1), 
                             torch.unsqueeze(dist_n, 1)))
        
        if train:
            # find cutoff parameter
            median_p = torch.median(dist_p).detach().cpu()
            median_n = torch.median(dist_n).detach().cpu()

            # we want median_p < median_n
            if median_p < median_n:
                new_cutoff = ((median_n - median_p)/2) + median_p
                self.cutoffs.append(new_cutoff)

                # moving average for updating cutoff value
                last_cut_offs = self.cutoffs[-self.moving_avg_len:]
                self.cutoff = np.mean(last_cut_offs)
                print(f"self.cutoff {self.cutoff}")

        # value BELOW cutoff is positive, above is negative
        # cutoff = 1.0 is okay
        result = torch.where(dist < self.cutoff, 1.0, 0.0) 

        ground_truth_p = (a_label == p_label).float()
        ground_truth_p = torch.unsqueeze(ground_truth_p, 1)

        ground_truth_n = (a_label == n_label).float()
        ground_truth_n = torch.unsqueeze(ground_truth_n, 1)

        ground_truth = torch.vstack((ground_truth_p, ground_truth_n))

        if visualise:
            print(f"dist: {torch.squeeze(dist)}")
            print(f"ground_truth: {torch.squeeze(ground_truth)}")

        acc = self.acc_criterion(result, ground_truth)

        return acc


    def training_step(self, batch, batch_idx):
        a_sample, a_label, a_dets, \
            p_sample, p_label, p_dets, \
                n_sample, n_label, n_dets = batch
        
        a_out = self(a_sample)
        p_out = self(p_sample)
        n_out = self(n_sample)

        loss = self.triplet_criterion(a_out, p_out, n_out)

        visualise = False
        if batch_idx == 0 and self.visualise:
            visualise = True

        acc = self.accuracy(a_out, p_out, n_out, a_label, p_label, n_label, visualise, train=True)

        self.log("train/cutoff", self.cutoff, on_step=True, on_epoch=True, batch_size=self.batch_size)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('train/acc', acc, on_step=True, on_epoch=True, batch_size=self.batch_size)

        if batch_idx == 99:
            print("train loss:", loss)
            print("train acc:", acc)

        return loss

    def evaluate(self, batch, name, stage=None):
        a_sample, a_label, a_dets, \
            p_sample, p_label, p_dets, \
                n_sample, n_label, n_dets = batch
        
        a_out = self(a_sample)
        p_out = self(p_sample)
        n_out = self(n_sample)

        loss = self.triplet_criterion(a_out, p_out, n_out)

        acc = self.accuracy(a_out, p_out, n_out, a_label, p_label, n_label, self.visualise, train=False)

        # TODO come up with a cutoff for when the distance is small enough to be called the same
        # for i in np.arange(len(ground_truth)):
        #     if ground_truth[i]:
        #         print(f"positive: {dist[i]}")
        #     else:
        #         print(f"negative: {dist[i]}")
        print(f"{stage}/{name} loss:", loss.detach().cpu().numpy())
        print(f"{stage}/{name} acc:", acc.detach().cpu().numpy())

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

    def on_save_checkpoint(self, checkpoint):
        # we update the cutoff parameter during training and want to save this new value
        checkpoint["learned_cutoff"] = self.cutoff

    def on_load_checkpoint(self, checkpoint):
        # load the learned cutoff parameter
        if "learned_cutoff" in checkpoint:
            self.cutoff = checkpoint["learned_cutoff"]
            self.cutoffs = [self.cutoff] * self.moving_avg_len