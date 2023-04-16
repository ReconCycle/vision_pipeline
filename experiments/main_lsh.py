import sys
import os
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
# import json
from rich import print
from PIL import Image
from tqdm import tqdm
import logging
from types import SimpleNamespace
import torch
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torchmetrics import Accuracy
from torchmetrics.classification import BinaryAccuracy

import argparse

import exp_utils as exp_utils

from data_loader import DataLoader

# https://github.com/guofei9987/pyLSHash

# from pyLSHash.pyLSHash.lshash import LSHash
from pyLSHash import LSHash
from model_lsh import LSHModel
from sklearn import preprocessing


class Main():
    def __init__(self, raw_args=None) -> None:
        exp_utils.init_seeds(1, cuda_deterministic=False)

        parser = argparse.ArgumentParser(
            description='pairwise classifier',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        seen_classes = ["hca_0", "hca_1", "hca_2", "hca_2a"]
        unseen_classes = ["hca_7", "hca_8", "hca_9", "hca_10"]
        # seen_classes = ["hca_0", "hca_1", "hca_2", "hca_2a", "hca_3", "hca_4", "hca_5", "hca_6"]
        # unseen_classes = ["hca_7", "hca_8", "hca_9", "hca_10", "hca_11", "hca_11a", "hca_12"]

        img_path = "experiments/datasets/2023-02-20_hca_backs"
        preprocessing_path = "experiments/datasets/2023-02-20_hca_backs_preprocessing_opencv"
        results_base_path = "experiments/results/"

        # for eval only with pairwise_classifier:
        results_path = "experiments/results/2023-03-14__17-38-43"
        checkpoint_path = "lightning_logs/version_0/checkpoints/epoch=19-step=2000.ckpt"

        parser.add_argument('--mode', type=str, default="train") # train/eval
        parser.add_argument('--model', type=str, default='lsh') # superglue/sift/pairwise_classifier
        parser.add_argument('--visualise', type=bool, default=False)
        parser.add_argument('--cutoff', type=float, default=0.01) # SIFT=0.01, superglue=0.5
        parser.add_argument('--batch_size', type=float, default=8)
        parser.add_argument('--batches_per_epoch', type=float, default=100)
        parser.add_argument('--train_epochs', type=float, default=30)
        parser.add_argument('--eval_epochs', type=float, default=1)
        parser.add_argument('--early_stopping', type=bool, default=True)
        # for pairwise_classifier only (during training):
        parser.add_argument('--freeze_backbone', type=bool, default=True)

        parser.add_argument('--img_path', type=str, default=img_path)
        parser.add_argument('--preprocessing_path', type=str, default=preprocessing_path)
        parser.add_argument('--results_base_path', type=str, default=results_base_path)
        parser.add_argument('--seen_classes', type=str, nargs='+', default=seen_classes)
        parser.add_argument('--unseen_classes', type=str, nargs='+', default=unseen_classes)

        # for eval only:
        parser.add_argument('--results_path', type=str, default=results_path)
        parser.add_argument('--checkpoint_path', type=str, default=checkpoint_path)

        self.args = parser.parse_args(raw_args)
        
        if self.args.mode == "train" or \
        (self.args.mode == "eval" and (self.args.model == "superglue" or  self.args.model == "sift")):
            # ignore checkpoint
            self.args.checkpoint_path = ""

            # create a new directory when we train
            results_path = os.path.join(self.args.results_base_path, datetime.now().strftime('%Y-%m-%d__%H-%M-%S' + f"_{self.args.model}"))
            if os.path.isdir(results_path):
                print("[red]results_path already exists!")
                return
            else:
                os.makedirs(results_path)
            self.args.results_path = results_path
        
        
        torch.set_grad_enabled(True)
        
        self.model = LSHModel(self.args.batch_size,
                                             freeze_backbone=self.args.freeze_backbone)

        self.lsh = LSHash(hash_size=6, input_dim=1000)
        
        print(f"self.lsh.num_hashtables {self.lsh.num_hashtables}") # 1
        print(f"len(self.lsh.uniform_planes) {len(self.lsh.uniform_planes)}") # 1


        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self.args.model == "lsh":
            print("using lsh")
        
        self.transform = transforms.Compose([
            # transforms.Resize((64, 64)),
            transforms.ToTensor(),
            # computed transform using compute_mean_std() to give:
            transforms.Normalize(mean=[0.5895, 0.5935, 0.6036],
                                std=[0.1180, 0.1220, 0.1092])
        ])
        
        self.dl = DataLoader(self.args.img_path,
                                    preprocessing_path=self.args.preprocessing_path,
                                    batch_size=self.args.batch_size,
                                    num_workers=8,
                                    shuffle=False, #! we want all of the same classes
                                    seen_classes=self.args.seen_classes,
                                    unseen_classes=self.args.unseen_classes,
                                    train_transform=self.transform,
                                    val_transform=self.transform,
                                    cuda=False)
        
        logging.basicConfig(filename=os.path.join(self.args.results_path, f'{self.args.mode}.log'), level=logging.DEBUG)
        self.log_args()

        # if self.args.mode == "fit":
        #     self.fit()
        # elif self.args.mode == "eval":
        #     self.eval()

        self.fit()
        self.eval()

    
    def log_args(self):
        for arg in vars(self.args):
            print(str(arg) + ": " + str(getattr(self.args, arg)))
            logging.info(str(arg) + ": " + str(getattr(self.args, arg)))

        print("device: " + str(self.device))
        logging.info("device: " + str(self.device))


    def fit(self):
        
        self.model.to(self.device)
        self.model.eval()
        
        print("fitting...")
        results = []
        unique_labels = set()
        for i, (sample, label, path, poly) in enumerate(tqdm(self.dl.dataloaders["seen_train"])):
            sample = sample.to(self.device)
            
            out = self.model(sample)

            np_label = label.cpu().detach().numpy()
            np_out = out.cpu().detach().numpy()

            # print("np_out", np_out.shape)

            # iterate over the batch
            # batch_result = []
            for i in np.arange(len(np_out)):
                # lsh.index([2, 3, 4, 5, 6, 7, 8, 9], extra_data="some vector info")
                unique_labels.add(np_label[i])
                # hash = self.lsh.index(np_out[i], extra_data=np_label[i])

                # hash = self.lsh._hash(self.lsh.uniform_planes[0], list(np_out[i]))
                projections = np.dot(self.lsh.uniform_planes[0], np_out[i])
                projections_bin = [1 if i > 0 else 0 for i in projections]

                # print(f"projections_bin {projections_bin}")
                results.append([projections_bin, np_label[i]])

                # print(f"hash {hash} {type(hash)}")

        # TODO: plot the distribution of the hash for each class
        print(f"unique_labels {unique_labels}")
        for label in unique_labels:
            print(f"label: {label}")
            results_for_label = [result[0] for result in results if result[1] == label]
            results_for_label = np.array(results_for_label)
            # print(f"results_for_label: {results_for_label}")
            num_results = len(results_for_label)
            hist = np.sum(results_for_label, axis=0) / num_results
            hist = preprocessing.normalize(hist) # 

            print(f"hist: {hist}")







    # https://github.com/aayushmnit/Deep_learning_explorations/blob/master/8_Image_similarity_search/Image%20similarity%20on%20Caltech101%20using%20FastAI%2C%20Pytorch%20and%20Locality%20Sensitive%20Hashing.ipynb
    def eval(self):
        
        self.model.to(self.device)
        self.model.eval()

        results = []
        unique_labels = set()
        for i, (sample, label, path, poly) in enumerate(tqdm(self.dl.dataloaders["unseen_test"])):
            
            sample = sample.to(self.device)
            
            out = self.model(sample)

            np_label = label.cpu().detach().numpy()
            np_out = out.cpu().detach().numpy()
        
            # print("")
            for i in np.arange(len(np_out)):
                # print(f"label: {np_label[i]}")
                unique_labels.add(np_label[i])


                projections = np.dot(self.lsh.uniform_planes[0], np_out[i])
                projections_bin = [1 if i > 0 else 0 for i in projections]

                # hash = self.lsh._hash(self.lsh.uniform_planes[0], list(np_out[i]))
                # print(f"hash eval: {projections_bin}")

                results.append([projections_bin, np_label[i]])

                # using query:
                # res = self.lsh.query(np_out[i], num_results=2)

                # # print(f"len(res): {len(res)}")
                # for ((vec, extra_data), distance) in res:
                #     print(f"len(vec): {len(vec)}, vec[0]: {vec[0]}, extra_data: {extra_data}, distance: {distance}")
                
                # # print(f"res: {res}")
                # results.append([res, np_label[i]])

        print(f"eval unique_labels {unique_labels}")
        # TODO: plot the distribution of the hash for each class
        for label in unique_labels:
            print(f"eval label: {label}")
            results_for_label = [result[0] for result in results if result[1] == label]
            results_for_label = np.array(results_for_label)
            # print(f"results_for_label: {results_for_label}")
            num_results = len(results_for_label)
            hist = np.sum(results_for_label, axis=0) / num_results

            print(f"eval hist: {hist}")

        # for label in unique_labels:
        #     results_for_label = [result[0] for result in results if result[1] == label]
        #     results_for_label = np.array(results_for_label)
        #     print(f"results_for_label: {results_for_label.shape}")


        # classes = self.dl.classes["unseen_test"]
        # print("classes", classes)

        # samples = self.dl.datasets["unseen_test"].targets
        # print("samples", samples)

        # for class_name in classes:
            
            



        
        
    # def eval_manual(self):
    #     self.results_path = "experiments/results/2023-03-10__15-28-03"
    #     self.checkpoint = "lightning_logs/version_0/checkpoints/epoch=19-step=2000.ckpt"
    #     self.model.load_from_checkpoint(os.path.join(self.results_path, self.checkpoint), strict=False)
    #     self.model.to(self.device)
    #     self.model.eval()
        
    #     for i, (sample1, label1, dets1, sample2, label2, dets2) in enumerate(self.dl.dataloaders["seen_train"]):
    #         sample1 = sample1.to(self.device)
    #         sample2 = sample2.to(self.device)
            
    #         out = self.model(sample1, sample2)
            
    #         print("out", out)

# https://github.com/Lightning-AI/lightning/issues/2110#issuecomment-1114097730
class OverrideEpochStepCallback(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def _log_step_as_current_epoch(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        pl_module.log("step", torch.tensor(trainer.current_epoch, dtype=torch.float32))
        

if __name__ == '__main__':
    main = Main()
    