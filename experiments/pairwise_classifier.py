import sys
import os
from datetime import datetime
import cv2
import numpy as np
import json
from rich import print
from PIL import Image
from tqdm import tqdm
import logging
from types import SimpleNamespace
import torch
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torchmetrics.classification import BinaryAccuracy

import exp_utils as exp_utils

from data_loader_even_pairwise import DataLoaderEvenPairwise

# do as if we are in the parent directory
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from graph_relations import GraphRelations
from object_reid import ObjectReId

from superglue.models.matching import Matching
from superglue.models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)


class PairwiseClassifier():
    def __init__(self) -> None:
        exp_utils.init_seeds(1, cuda_deterministic=False)
        self.args = SimpleNamespace()
        self.args.visualise = True
        self.args.cutoff = 0.5
        self.args.model = "superglue"
        self.args.img_path = "experiments/datasets/2023-02-20_hca_backs"
        self.args.preprocessing_path = "experiments/datasets/2023-02-20_hca_backs_preprocessing_opencv"
        self.args.results_base_path = "experiments/results/"
        # seen_classes = ["hca_0", "hca_1", "hca_2", "hca_2a", "hca_3", "hca_4", "hca_5", "hca_6"]
        # unseen_classes = ["hca_7", "hca_8", "hca_9", "hca_10", "hca_11", "hca_11a", "hca_12"]
        
        self.args.seen_classes = ["hca_0", "hca_1", "hca_2" , "hca_3", "hca_4", "hca_5", "hca_6", "hca_7", "hca_8", "hca_9", "hca_10", "hca_11", "hca_12"]
        self.args.unseen_classes = ["hca_2a", "hca_11a"]
        self.args.batch_size = 8
        
        torch.set_grad_enabled(True)
        
        opt = SimpleNamespace()
        opt.superglue = "indoor"
        opt.nms_radius = 4
        opt.sinkhorn_iterations = 20
        opt.match_threshold = 0.5 # default 0.2
        opt.show_keypoints = True
        opt.keypoint_threshold = 0.005
        opt.max_keypoints = -1
        
        self.matching_config = {
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
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        #! LOAD WEIGHTS FOR BACKBONE MODEL
        self.matching = Matching(self.matching_config).to(self.device)
        self.model = PairWiseClassifierModel(superpoint=self.matching.superpoint)
        
        
        self.dl = DataLoaderEvenPairwise(self.args.img_path,
                                    preprocessing_path=self.args.preprocessing_path,
                                    batch_size=self.args.batch_size,
                                    num_workers=2,
                                    shuffle=True,
                                    seen_classes=self.args.seen_classes,
                                    unseen_classes=self.args.unseen_classes)
    
    
    def train(self):
        self.results_path = os.path.join(self.args.results_base_path, datetime.now().strftime('%Y-%m-%d__%H-%M-%S'))
        if os.path.isdir(self.results_path):
            print("[red]results_path already exists!")
        else:
            os.makedirs(self.results_path)
        
        logging.basicConfig(filename=os.path.join(self.results_path, 'train.log'), level=logging.DEBUG)
        
        logging.info("visualise: " + str(self.args.visualise))
        logging.info("cutoff: " + str(self.args.cutoff))
        logging.info("model: " + str(self.args.model))
        logging.info("img_path: " + str(self.args.img_path))
        logging.info("preprocessing_path: " + str(self.args.preprocessing_path))
        logging.info("seen_classes: " + str(self.args.seen_classes))
        logging.info("unseen_classes: " + str(self.args.unseen_classes))
        logging.info("device: " + str(self.device))
        
        trainer = pl.Trainer(
            default_root_dir=self.results_path,
            limit_train_batches=100,
            limit_val_batches=100,
            limit_test_batches=100,
            max_epochs=20,
            accelerator="gpu",
            devices=1)
        trainer.fit(model=self.model, train_dataloaders=self.dl.dataloaders["seen_train"])

    def eval(self):
        self.results_path = "experiments/results/2023-03-09__12-56-10"
        self.checkpoint = "lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
        self.model.load_from_checkpoint(os.path.join(self.results_path, self.checkpoint), strict=False)

        # disable randomness, dropout, etc...
        # model.eval()
        
        # predict with the model
        # y_hat = model(x)

        # OR:
        trainer = pl.Trainer(default_root_dir=self.results_path,
                            limit_train_batches=100,
                            limit_val_batches=100,
                            limit_test_batches=100,
                            max_epochs=1,
                            accelerator='gpu',
                            devices=1)

        # test the model
        trainer.test(self.model, dataloaders=self.dl.dataloaders["seen_train"])
        
    def eval_manual(self):
        self.results_path = "experiments/results/2023-03-09__12-56-10"
        self.checkpoint = "lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
        self.model.load_from_checkpoint(os.path.join(self.results_path, self.checkpoint), strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        for i, (sample1, label1, dets1, sample2, label2, dets2) in enumerate(self.dl.dataloaders["seen_train"]):
            sample1 = sample1.to(self.device)
            sample2 = sample2.to(self.device)
            
            out = self.model(sample1, sample2)
            
            print("out", out)


# define the LightningModule
class PairWiseClassifierModel(pl.LightningModule):
    def __init__(self, model=None, superpoint=None):
        super().__init__()
        self.model = model
        self.superpoint= superpoint
        if model is None:
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
            self.model.requires_grad = True
            for param in self.model.parameters():
                param.requires_grad = True
            # https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/cifar10-baseline.html
            # ? maybe I should end with F.log_softmax(out, dim=1)
            # ? then do: loss = F.nll_loss(logits, y)

    def forward(self, sample1, sample2):
        # we want to freeze the backbone
        # self.superpoint.eval()
        # with torch.no_grad():
        img1_data = self.superpoint({'image': sample1})
        img2_data = self.superpoint({'image': sample2})
        
        x1 = img1_data["x_waypoint"] # shape: (1, 65, 50, 50)
        x2 = img2_data["x_waypoint"] # shape: (1, 65, 50, 50)
        
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
        
        if batch_idx == 99:
            print("x_out", x_out)
            print("ground truth", ground_truth)
        
        
        # loss = nn.functional.mse_loss(x_out, ground_truth)
        criterion = nn.BCEWithLogitsLoss() # This loss combines a Sigmoid layer and BCELoss in one class
        loss = criterion(x_out, ground_truth)
        self.log("train_loss", loss, on_epoch=True)
        return loss


    def test_step(self, batch, batch_idx):
        # this is the test loop
        
        sample1, label1, dets1, sample2, label2, dets2 = batch
        
        ground_truth = (label1 == label2).float()
        ground_truth = torch.unsqueeze(ground_truth, 1)
        
        # run the forward step
        x_out = self(sample1, sample2)
        
        test_loss = nn.functional.mse_loss(x_out, ground_truth)
        self.log("test_loss", test_loss)
        
        #! How should binary accuracy work?
        #! We should set a cut off
        accuracy = BinaryAccuracy().to(self.device)
        # accuracy = Accuracy(task="binary").to(self.device)
        acc = accuracy(x_out, ground_truth)
        self.log('accuracy', acc, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-5) # was 1e-3
        return optimizer


if __name__ == '__main__':
    pairwise_classifier = PairwiseClassifier()
    pairwise_classifier.train()
    # pairwise_classifier.eval()
    # pairwise_classifier.eval_manual()
    