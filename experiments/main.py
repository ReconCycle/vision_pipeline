import sys
import os
from datetime import datetime
import cv2
import numpy as np
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
import albumentations as A
from albumentations.pytorch import ToTensorV2

import argparse

import exp_utils as exp_utils
from exp_utils import str2bool

from data_loader_even_pairwise import DataLoaderEvenPairwise
from data_loader_triplet import DataLoaderTriplet

from model_pairwise_classifier import PairWiseClassifierModel
from model_pairwise_classifier2 import PairWiseClassifier2Model
from model_pairwise_classifier3 import PairWiseClassifier3Model
from model_triplet import TripletModel
from model_superglue import SuperGlueModel
from model_sift import SIFTModel
from model_clip import ClipModel
from model_cosine import CosineModel

# do as if we are in the parent directory
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from graph_relations import GraphRelations
from object_reid import ObjectReId

from superglue.models.matching import Matching
from superglue.models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)    


class Main():
    def __init__(self, raw_args=None) -> None:
        exp_utils.init_seeds(1, cuda_deterministic=False)

        parser = argparse.ArgumentParser(
            description='pairwise classifier',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # seen_classes = ["hca_0", "hca_1", "hca_2", "hca_2a"]
        # unseen_classes = ["hca_7", "hca_8", "hca_9", "hca_10"]
        seen_classes = ["hca_0", "hca_1", "hca_2", "hca_2a", "hca_3", "hca_4", "hca_5", "hca_6"]
        unseen_classes = ["hca_7", "hca_8", "hca_9", "hca_10", "hca_11", "hca_11a", "hca_12"]

        base_path = os.path.dirname(os.path.abspath(__file__))
        print("base path:", base_path)

        img_path = os.path.join(base_path, "datasets/2023-02-20_hca_backs")
        preprocessing_path = os.path.join(base_path, "datasets/2023-02-20_hca_backs_preprocessing_opencv")
        results_base_path = os.path.join(base_path, "results/")

        # for eval only with pairwise_classifier:
        results_path = os.path.join(base_path, "results/2023-03-14__17-38-43")
        checkpoint_path = "lightning_logs/version_0/checkpoints/epoch=19-step=2000.ckpt"

        eval_only_models = ["superglue", "cosine", "clip"]

        parser.add_argument('--mode', type=str, default="train") # train/eval
        parser.add_argument('--model', type=str, default='triplet') # superglue/sift/pairwise_classifier/pairwise_classifier2/triplet
        parser.add_argument('--visualise', type=str2bool, default=False)
        parser.add_argument('--cutoff', type=float, default=0.01) # SIFT=0.01, superglue=0.5
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--batches_per_epoch', type=int, default=300)
        parser.add_argument('--train_epochs', type=int, default=50)
        parser.add_argument('--eval_epochs', type=int, default=1)
        parser.add_argument('--early_stopping', type=str2bool, default=True)
        # , nargs='?', const=True,
        # for pairwise_classifier only (during training):
        parser.add_argument('--freeze_backbone', type=str2bool, default=True)
        parser.add_argument('--learning_rate', type=float, default=1e-5)
        parser.add_argument('--weight_decay', type=float, default=0.0)

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
        (self.args.mode == "eval" and (self.args.model in eval_only_models)):
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
        
        opt = SimpleNamespace()
        opt.superglue = "indoor"
        opt.nms_radius = 4
        opt.sinkhorn_iterations = 20
        opt.match_threshold = 0.5 # default 0.2
        opt.show_keypoints = True
        opt.keypoint_threshold = 0.005
        opt.max_keypoints = -1
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model_rgb = True
        if self.args.model in ["superglue", "sift"]:
            self.model_rgb = False

        val_tf_list = []
        train_tf_list = [
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.25, rotate_limit=45, p=0.5),
            A.OpticalDistortion(p=0.5),
            A.GridDistortion(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.RandomResizedCrop(400, 400, p=0.3),
            A.RGBShift(r_shift_limit=40, g_shift_limit=40, b_shift_limit=40, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
        ]

        # computed transform using compute_mean_std() to give:
        # transform_normalise = A.Normalize(mean=(0.5895, 0.5935, 0.6036), std=(0.1180, 0.1220, 0.1092))
        # imagenet normalize
        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]
        transform_normalise = A.Normalize(mean=self.norm_mean, std=self.norm_std) # imagenet
        transform_resize = A.augmentations.geometric.resize.LongestMaxSize(max_size=224, always_apply=True)

        if self.model_rgb:
            # validation transforms
            val_tf_list.append(transform_normalise)
            if self.args.model == "clip":
                val_tf_list.append(transform_resize)
            val_tf_list.append(ToTensorV2())
        
            # train transforms
            train_tf_list.append(transform_normalise)    
            if self.args.model == "clip":
                train_tf_list.append(transform_resize)
            train_tf_list.append(ToTensorV2())

            self.val_transform = A.Compose(val_tf_list)
            self.train_transform = A.Compose(train_tf_list)
        else:
            # don't normalise. We only evaluate these models
            self.val_transform = A.Compose([
                ToTensorV2()
            ])

            train_tf_list.append(ToTensorV2())
            self.train_transform = A.Compose(train_tf_list)

        # setup model
        if self.args.model == "superglue":
            print("using SuperGlueModel")
            if self.args.mode == "train":
                print("[red]This model is eval only![/red]")
                return
            self.model = SuperGlueModel(self.args.batch_size, opt=opt, visualise=self.args.visualise)
        elif self.args.model == "sift":
            print("using SIFT")
            self.model = SIFTModel(self.args.batch_size, cutoff=self.args.cutoff, visualise=self.args.visualise)

        elif self.args.model == "pairwise_classifier":
            print("using pairwise_classifier")
            self.model = PairWiseClassifierModel(self.args.batch_size, 
                                                 self.args.learning_rate, 
                                                 self.args.weight_decay,
                                                 opt=opt, 
                                                 freeze_backbone=self.args.freeze_backbone)
            
        elif self.args.model == "pairwise_classifier2":
            print("using pairwise_classifier2")
            self.model = PairWiseClassifier2Model(self.args.batch_size,
                                                  self.args.learning_rate,
                                                  self.args.weight_decay,
                                                  freeze_backbone=self.args.freeze_backbone,
                                                  visualise=self.args.visualise)
        elif self.args.model == "pairwise_classifier3":
            print("using pairwise_classifier3")
            self.model = PairWiseClassifier3Model(self.args.batch_size,
                                                  self.args.learning_rate,
                                                  self.args.weight_decay,
                                                  cutoff=self.args.cutoff, 
                                                  freeze_backbone=self.args.freeze_backbone,
                                                  visualise=self.args.visualise)
        elif self.args.model == "cosine":
            print("using cosine model")
            self.model = CosineModel(self.args.batch_size, 
                                   cutoff=self.args.cutoff, 
                                   visualise=self.args.visualise)
            
        elif self.args.model == "clip":
            self.model = ClipModel(self.args.batch_size, 
                                   cutoff=self.args.cutoff, 
                                   visualise=self.args.visualise)
        elif self.args.model == "triplet":
            print("using triplet")
            self.model = TripletModel(self.args.batch_size,
                                      learning_rate=self.args.learning_rate,
                                      weight_decay=self.args.weight_decay,
                                      cutoff=self.args.cutoff,
                                      freeze_backbone=self.args.freeze_backbone,
                                      visualise=self.args.visualise)

        
        # setup dataloader
        if self.args.model in ["superglue", "sift", "pairwise_classifier", "pairwise_classifier2", "pairwise_classifier3", "cosine", "clip"]:
            self.dl = DataLoaderEvenPairwise(self.args.img_path,
                                        preprocessing_path=self.args.preprocessing_path,
                                        batch_size=self.args.batch_size,
                                        num_workers=8,
                                        shuffle=True,
                                        seen_classes=self.args.seen_classes,
                                        unseen_classes=self.args.unseen_classes,
                                        train_transform=self.train_transform,
                                        val_transform=self.val_transform)
        elif self.args.model in ["triplet"]:
            self.dl = DataLoaderTriplet(self.args.img_path,
                                        preprocessing_path=self.args.preprocessing_path,
                                        batch_size=self.args.batch_size,
                                        num_workers=8,
                                        shuffle=True,
                                        seen_classes=self.args.seen_classes,
                                        unseen_classes=self.args.unseen_classes,
                                        train_transform=self.train_transform,
                                        val_transform=self.val_transform)

        
        logging.basicConfig(filename=os.path.join(self.args.results_path, f'{self.args.mode}.log'), level=logging.DEBUG)
        self.log_args()

        self.dl.dataloader_imgs.visualise("seen_train", mean=self.norm_mean, std=self.norm_std, save_path=self.args.results_path)

        if self.args.mode == "train":
            self.train()
        elif self.args.mode == "eval":
            self.eval()

    
    def log_args(self):
        for arg in vars(self.args):
            print(f"{arg}: {getattr(self.args, arg)}")
            logging.info(f"{arg}: {getattr(self.args, arg)}")

        print(f"device: {self.device}")
        logging.info(f"device: {self.device}")

        print(f"train transforms: {self.train_transform}")
        logging.info(f"train transforms: {self.train_transform}")

        print(f"val transforms: {self.val_transform}")
        logging.info(f"val transforms: {self.val_transform}")

        print(f"model: {self.model}")
        logging.info(f"model: {self.model}")

    def train(self):
        
        logging.info(f"starting training...")

        callbacks = [OverrideEpochStepCallback()]
        checkpoint_callback = ModelCheckpoint(monitor="val/seen_val/loss_epoch", mode="max", save_top_k=1)
        callbacks.append(checkpoint_callback)
        if self.args.early_stopping:
            early_stop_callback = EarlyStopping(monitor="val/seen_val/loss_epoch", mode="max", patience=10, verbose=False, strict=True)
            callbacks.append(early_stop_callback)
            

        trainer = pl.Trainer(
            callbacks=callbacks,
            enable_checkpointing=True,
            default_root_dir=self.args.results_path,
            limit_train_batches=self.args.batches_per_epoch,
            limit_val_batches=self.args.batches_per_epoch,
            limit_test_batches=self.args.batches_per_epoch,
            max_epochs=self.args.train_epochs,
            accelerator="gpu",
            devices=1)

        self.model.val_datasets = ["seen_val"]
        trainer.fit(model=self.model, 
                    train_dataloaders=self.dl.dataloaders["seen_train"],
                    val_dataloaders=self.dl.dataloaders["seen_val"])
        

        logging.info(f"best model path: {checkpoint_callback.best_model_path}")
        logging.info(f"best model score: {checkpoint_callback.best_model_score}")
        print(f"best model path: {checkpoint_callback.best_model_path}")
        print(f"best model score: {checkpoint_callback.best_model_score}")

        # immediately run eval
        self.eval(model_path=checkpoint_callback.best_model_path)
        

    def eval(self, model_path=None):
        logging.info(f"running eval...")

        if model_path is None and self.args.model in ["pairwise_classifier", "pairwise_classifier2", "pairwise_classifier3", "triplet"]:
            model_path = os.path.join(self.args.results_path, self.args.checkpoint_path)
        
        print(f"model_path {model_path}")
        logging.info(f"model_path {model_path}")

        if self.args.model == "pairwise_classifier":
            self.model = PairWiseClassifierModel.load_from_checkpoint(model_path, strict=False)
            print("loaded checkpoint!")
        elif self.args.model == "pairwise_classifier2":
            self.model = PairWiseClassifier2Model.load_from_checkpoint(model_path, strict=False)
            print("loaded checkpoint!")
        elif self.args.model == "pairwise_classifier3":
            self.model = PairWiseClassifier3Model.load_from_checkpoint(model_path, strict=False)
            print("loaded checkpoint!")
        elif self.args.model == "triplet":
            self.model = TripletModel.load_from_checkpoint(model_path, strict=False)
            print("loaded checkpoint!")

        
        trainer = pl.Trainer(callbacks=[OverrideEpochStepCallback()],
                            default_root_dir=self.args.results_path,
                            limit_train_batches=self.args.batches_per_epoch,
                            limit_val_batches=self.args.batches_per_epoch,
                            limit_test_batches=self.args.batches_per_epoch,
                            max_epochs=self.args.eval_epochs,
                            accelerator='gpu',
                            devices=1)

        # test the model
        print("[blue]eval_results:[/blue]")
        test_datasets = ["seen_train", "seen_val", "test"]
        self.model.test_datasets = test_datasets
        output = trainer.test(self.model,
                                  dataloaders=[self.dl.dataloaders[name] for name in test_datasets])
        for i, name in enumerate(test_datasets):
            logging.info(f"eval {name}:" + str(output[i]))

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
# logging: https://github.com/Lightning-AI/lightning/discussions/6182
class OverrideEpochStepCallback(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        metrics = trainer.callback_metrics
        
        # print("on_train_epoch_end metrics", metrics)
        train_info = "train (epoch " + str(trainer.current_epoch) + "): "
        if 'train/loss_epoch' in metrics:
            train_info += str(metrics["train/loss_epoch"].cpu().numpy()) + ", "
        
        if 'train/acc_epoch' in metrics:
            train_info += str(metrics["train/acc_epoch"].cpu().numpy())

        logging.info(train_info)
        print(train_info)

        self._log_step_as_current_epoch(trainer, pl_module)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        metrics = trainer.callback_metrics
        # print("on_test_epoch_end metrics", metrics)

        test_info = "test (epoch " + str(trainer.current_epoch) + "): "
        if 'test/test/loss_epoch' in metrics:
            test_info += str(metrics["test/test/loss_epoch"].cpu().numpy()) + ", "
        
        if "test/test/acc_epoch" in metrics:
            test_info += str(metrics["test/test/acc_epoch"].cpu().numpy())

        logging.info(test_info)
        print(test_info)

        self._log_step_as_current_epoch(trainer, pl_module)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):

        metrics = trainer.callback_metrics
        # print("on_validation_epoch_end metrics", metrics)

        valid_info = "valid (epoch " + str(trainer.current_epoch) + "): "
        if 'val/seen_val/loss_epoch' in metrics:
            valid_info += str(metrics["val/seen_val/loss_epoch"].cpu().numpy()) + ", "
        
        if "val/seen_val/acc_epoch" in metrics:
            valid_info += str(metrics["val/seen_val/acc_epoch"].cpu().numpy())

        logging.info(valid_info)
        print(valid_info)
    
        self._log_step_as_current_epoch(trainer, pl_module)

    def _log_step_as_current_epoch(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        pl_module.log("step", torch.tensor(trainer.current_epoch, dtype=torch.float32))
        

if __name__ == '__main__':
    main = Main()
    