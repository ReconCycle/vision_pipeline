from torchvision import datasets
import torch
# from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
# import albumentations as A
# import albumentations.pytorch
import time
import numpy as np
import cv2
import os
import regex as re
from tqdm import tqdm
import math
from collections import Counter
import sys
from rich import print
import exp_utils as exp_utils

from data_loader import DataLoader

# triplet inspiration
# https://www.kaggle.com/code/hirotaka0122/triplet-loss-with-pytorch/notebook


class TripletDataset(torch.utils.data.Dataset):
    """Gives an even split of sample pairs with the same label, as sample pairs with different labels

    Args:
        torch (dataset): dataset of items providing (sample, label, _*)
    """
    def __init__(self, dataset, train=True):    
        self.dataset = dataset
        self.train = train

        # n is number of items
        self.n = len(dataset)
        # print("self.n", self.n)

        self.dataset_labels = [label for _, label, *_ in dataset]
        # print(f"dataset_labels len {len(self.dataset_labels)}")

            
    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        
        # anchor
        a_sample, a_label, path, a_dets = self.dataset[idx]

        if self.train:
            pos_list = [idx for idx, dataset_label in enumerate(self.dataset_labels) if a_label == dataset_label]

            neg_list = [idx for idx, dataset_label in enumerate(self.dataset_labels) if a_label != dataset_label]

            # print(f"pos_list {pos_list}")

            pos_idx = pos_list[torch.randint(len(pos_list), (1,))]
            neg_idx = neg_list[torch.randint(len(neg_list), (1,))]

            p_sample, p_label, _, p_dets = self.dataset[pos_idx]
            n_sample, n_label, _, n_dets = self.dataset[neg_idx]
            
            return a_sample, a_label, a_dets, \
                    p_sample, p_label, p_dets, \
                    n_sample, n_label, n_dets

        else: # eval
            # sample positive or negative with 50/50
            rand_bool = torch.randint(2, (1,))[0].bool() # 0 or 1
            if rand_bool:
                # get positives
                sample_list = [idx for idx, dataset_label in enumerate(self.dataset_labels) if a_label == dataset_label]
            else:
                # get negatives
                sample_list = [idx for idx, dataset_label in enumerate(self.dataset_labels) if a_label != dataset_label]

            # randomly sample
            sample_idx = sample_list[torch.randint(len(sample_list), (1,))]

            sample2, label2, _, dets2 = self.dataset[sample_idx]

            return a_sample, a_label, a_dets, \
                    sample2, label2, dets2
            

    
class DataLoaderTriplet():
    """dataloader for EvenPairwiseDataset
    """
    def __init__(self,
                 img_path="MNIST",
                 preprocessing_path=None,
                 batch_size=256,
                 num_workers=8,
                 shuffle=True,
                 validation_split=.2,
                 seen_classes=[],
                 unseen_classes=[],
                 train_transform=None,
                 val_transform=None,
                 cuda=True):
        
        start = time.time()

        if img_path == "MNIST":
            raise NotImplementedError

        else:
            self.dataloader_imgs = DataLoader(img_path,
                                         preprocessing_path=preprocessing_path,
                                         batch_size=batch_size,
                                         shuffle=shuffle,
                                         validation_split=validation_split,
                                         seen_classes=seen_classes,
                                         unseen_classes=unseen_classes,
                                         limit_imgs_per_class=30, #! why limit?
                                         train_transform=train_transform,
                                         val_transform=val_transform,
                                         cuda=cuda)

        self.classes = self.dataloader_imgs.classes
        
        

        # ! to provide triplets only for seen_train, use:
        # ! train=x=="seen_train"
        self.dataloaders = {x: torch.utils.data.DataLoader(
                                    TripletDataset(self.dataloader_imgs.datasets[x], train=True),
                                    batch_size=batch_size,
                                    shuffle=shuffle,
                                    num_workers=num_workers,
                                    # collate_fn=custom_collate
                                )
                            for x in ["seen_train", "seen_val", "seen_test", "unseen_test", "test"]}
        
        end = time.time()
        print(f"[blue]elapsed time: {end - start}")
        
    def example(self):
        positive_pairs = 0
        negative_pairs = 0
        label_distribution = {}
        
        # classes_seen_train = self.classes["seen_train"]
        for a_class in self.classes["seen_train"]:
            label_distribution[a_class] = 0

        print("classes seen_train", self.classes["seen_train"])

        for i, (sample1, label1, dets1, sample2, label2, dets2) in enumerate(self.dataloaders["seen_train"]):
            pass
        
        print("")
        print("distributions:")
        print("\nbatch num:", i)
        print("positive_pairs", positive_pairs)
        print("negative_pairs", negative_pairs)
        print("label_distribution", label_distribution)
        
        
        
if __name__ == '__main__':
    print("Run this for testing the dataloader only.")
    # img_path = "/home/sruiz/datasets2/reconcycle/simon_rgbd_dataset/hca_simon/sorted_in_folders"
    # img_path = "/home/sruiz/datasets/labelme/hca_front_21-10-01/cropped"
    # img_path = "MNIST"
    img_path = "experiments/datasets/2023-02-20_hca_backs"
    preprocessing_path = "experiments/datasets/2023-02-20_hca_backs_preprocessing_opencv"
    seen_classes = ["hca_0", "hca_1", "hca_2", "hca_2a", "hca_3", "hca_4", "hca_5", "hca_6"]
    unseen_classes = ["hca_7", "hca_8", "hca_9", "hca_10", "hca_11", "hca_11a", "hca_12"]
    
    exp_utils.init_seeds(1, cuda_deterministic=False)
    dataloader = DataLoaderTriplet(img_path,
                                        preprocessing_path=preprocessing_path,
                                        batch_size=32,
                                        shuffle=True,
                                        seen_classes=seen_classes,
                                        unseen_classes=unseen_classes)
    dataloader.example()