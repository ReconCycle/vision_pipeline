from torchvision import datasets
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time
import numpy as np
import cv2
import os
import regex as re
from tqdm import tqdm
import math
from collections import Counter
import sys
import itertools
import functools
import operator
from rich import print
import exp_utils as exp_utils

from data_loader import DataLoader
# from data_loader_mnist import DataLoaderMNIST

# inspiration:
# https://discuss.pytorch.org/t/how-to-take-pairwise-input-image-after-combing-the-models/124402/5
# https://colab.research.google.com/github/rramosp/2021.deeplearning/blob/master/content/U2%20LAB%2003%20-%20Pairwise%20image%20classification.ipynb#scrollTo=fU3h0AWM3ChA


def nC2(n):
    return int((n*(n-1))/2)

def pair(n, k):
    """Returns the pair given the number of items, from the total number of pairs nC2
    
    See: https://math.stackexchange.com/questions/4046051/how-to-find-the-nth-pair-from-a-generated-list-of-pairs
    
    The total number of pairs is nC2.
    Pairs are ordered as:
    (0,1),(0,2),...,(0,n-1),(1,2),(1,3),...,(1,n-1),...,(n-2, n-1)

    Args:
        n (int): Number of items
        k (int): Pair id. In range 0, ..., (nC2)-1

    Returns:
        int, int: the pair
    """
    # if k > nC2(n) - 1:
        # raise SystemExit("k out of range", k, nC2(n) - 1)
    
    # inside = 4*n*n-4*n+1-8*k
    # if inside < 0:
    #     raise SystemExit("negative value in sqrt", inside, "k", k, "n", n)
    x = math.floor((2*n-1-math.sqrt(4*n*n-4*n+1-8*k))/2)
    y = k - x*n + x*(x+1)//2 + x + 1
    return x, y

def pair_idx(n, x, y):
    """Inverse of pair(n, k). Given a pair it returns the index

    Args:
        n (int): number of items
        x (int): item x. In range 0, ..., n-1
        y (int): item y. In range 0, ..., n-1

    Returns:
        int: pair index. In range 0, ..., (nC2)-1
    """
    if x == y:
        return None
    if x > y:
        x, y = y, x
    
    k = y-(x+1) + x*n-((x*(x+1))/2)
    return k


def nth_item(n, item, iterable):
    """find the index of the n'th occurrence of an item in a list
    source: https://stackoverflow.com/questions/8337069/find-the-index-of-the-nth-item-in-a-list

    Args:
        n (int): nth occurance
        item (int): of this item
        iterable (list): in this list

    Returns:
        int: the index of the n'th occurance of the item in the list
    """
    indicies = itertools.compress(itertools.count(), map(functools.partial(operator.eq, item), iterable))
    return next(itertools.islice(indicies, n, None), -1)
    


class EvenPairwiseDataset(torch.utils.data.Dataset):
    """Gives an even split of sample pairs with the same label, as sample pairs with different labels

    Args:
        torch (dataset): dataset of items providing (sample, label, _*)
    """
    def __init__(self, dataset):
        # to get the dataset from a dataloader, use: dataloader.dataset
    
        self.dataset = dataset
        # n is number of items
        self.n = len(dataset)
        # print("self.n", self.n)
        self.dataset_labels = [label for _, label, *_ in dataset]
        self.samples_per_class = Counter(self.dataset_labels)
        # print("samples_per_class", self.samples_per_class)
        
        
        # compute total number of positive pairs:
        self.pairs_per_class = {}
        self.num_positive_pairs = 0
        for class_id, num_samples in self.samples_per_class.items():
            # print("samples", num_samples)
            num_pairs = nC2(num_samples)
            # print("num_pairs")
            self.pairs_per_class[class_id] = num_pairs
            self.num_positive_pairs += num_pairs
            
    def __len__(self):
        # we want 50/50 split of positive/negative pairs
        # the dataset length can't exceed nC2
        return np.minimum(2*self.num_positive_pairs, nC2(self.n))

    def __getitem__(self, idx):
        # we first index over all the negative pairs, then over all the positive pairs
        # print("")
        # print("self.n", self.n)
        # print("self.samples_per_class", self.samples_per_class)
        # print("self.pairs_per_class", self.pairs_per_class)
        # print("self.num_positive_pairs", self.num_positive_pairs)
        # print("")
        # print("idx", idx)
        id1 = -1
        id2 = -1
        
        sum = 0
        positive_pair = False
        
        # start = time.time()
        
        for class_id, num_pairs in self.pairs_per_class.items():
        # for i in np.arange(len(self.pairs_per_class)):
            if idx < sum + num_pairs:
                # print("positive pair")
                positive_pair = True
                # we are in this pair
                rel_idx = idx - sum
                # print("class_id", class_id)
                # print("num_pairs", num_pairs)
                # print("rel_idx", rel_idx)
                
                x1, x2 = pair(self.samples_per_class[class_id], rel_idx) # from n items, get the kth pair
                
                # print("x1", x1)
                # print("x2", x2)
                
                # the dataset isn't ordered, so we need to find the 'x1'th sample with label 'class_id' in the dataset
                
                id1 = nth_item(x1, class_id, self.dataset_labels)
                id2 = nth_item(x2, class_id, self.dataset_labels)
                break
            else:
                sum += num_pairs
        
        if not positive_pair:
            # this SHOULD be the same as: if idx > self.num_positive_pairs:
            # print("self.num_positive_pairs", self.num_positive_pairs)
            # print("should generate negative pair, or generate with high probability")
            # most pairs are negative, so let's return a pair
            # ? there is probably a way to guarantee we get a negative pair, but close enough for now
            id1, id2 = pair(self.n, idx)
            
        # print("id1", id1)
        # print("id2", id2)
        
        if id1 == -1 or id2 == -1:
            raise SystemExit("id1 or id2 is negative")
        
        sample1, label1, path1, detections1 = self.dataset[id1]
        sample2, label2, path2, detections2 = self.dataset[id2]
        
        # end = time.time()
        # print("__getitem__ elapsed time:", end - start)
                
        # print("label1", label1)
        # print("label2", label2)
        # print("")
        
        
        if positive_pair and label1 != label2:
            raise SystemExit("the labels of the pair should be the same, but they aren't")
        
        return sample1, label1, detections1, sample2, label2, detections2
    

    
class DataLoaderEvenPairwise():
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
        
        if img_path == "MNIST":
            raise NotImplementedError
            # self.dataloader_imgs = DataLoaderMNIST(batch_size=batch_size,
            #                                   shuffle=shuffle,
            #                                   validation_split=validation_split,
            #                                   seen_classes=seen_classes,
            #                                   unseen_classes=unseen_classes)

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

        def custom_collate(instances):
            # handle sample, label, and path like normal
            # but handle detections as list of lists.
            
            # elem = instances[0] # tuple: (sample1, label1, detections1, sample2, label2, detections2)
            
            batch = []
            
            for i in range(len(instances[0])):
                batch.append([instance[i] for instance in instances])
            
            # apply default collate for: sample1, label1, ..., sample2, label2
            # print(f"batch[0] {np.array(batch[0]).shape}")
            batch[0] = torch.utils.data.default_collate(batch[0])
            batch[1] = torch.utils.data.default_collate(batch[1])
            
            batch[3] = torch.utils.data.default_collate(batch[3])
            batch[4] = torch.utils.data.default_collate(batch[4])

            return batch

        self.classes = self.dataloader_imgs.classes
        self.dataloaders = {x: torch.utils.data.DataLoader(
                                    EvenPairwiseDataset(self.dataloader_imgs.datasets[x]),
                                    batch_size=batch_size,
                                    shuffle=shuffle,
                                    num_workers=num_workers,
                                    collate_fn=custom_collate
                                )
                            for x in ["seen_train", "seen_val", "seen_test", "unseen_test", "test"]}
        
    def example(self, type_name):
        positive_pairs = 0
        negative_pairs = 0
        label_distribution = {}
        
        # classes_seen_train = self.classes["seen_train"]
        for a_class in self.classes[type_name]:
            label_distribution[a_class] = 0

        print("classes seen_train", self.classes[type_name])

        for i, (sample1, label1, dets1, sample2, label2, dets2) in enumerate(self.dataloaders[type_name]):
            # print("\ni", i)
            # print("i % 100", i % 100)
            # print("sample1.shape", sample1.shape)
            # print("sample2.shape", sample2.shape)
            # print("label1.shape", label1.shape)
            # print("label2.shape", label2.shape)
            label1 = label1.detach().numpy()
            label2 = label2.detach().numpy()
            label1 = [self.classes[type_name][label] for label in label1]
            label2 = [self.classes[type_name][label] for label in label2]

            def get_img(sample):
                # opencv wants the channels to be first
                # sample = torch.einsum('cwh->whc', sample)
                # img = sample.detach().cpu().numpy()
                # img = (img * 255).astype(dtype=np.uint8)
                img = sample
                print("img.shape", img.shape)
                # img = np.squeeze(img, axis=0)
                return img


            img1 = get_img(sample1[0])
            img2 = get_img(sample2[0])

            vis_imgs = np.concatenate((img1, img2), axis=1)

            print("img1.shape", img1.shape)
            cv2.imshow("vis_imgs", vis_imgs)
            k = cv2.waitKey(0)

            for j in np.arange(len(label1)):
                if label1[j] == label2[j]:
                    positive_pairs += 1
                else:
                    negative_pairs += 1
                    
                if label1[j] in label_distribution.keys():
                    label_distribution[label1[j]] += 1
                else:
                    label_distribution[label1[j]] = 0
                    
            for j in np.arange(len(label2)):
                if label2[j] in label_distribution.keys():
                    label_distribution[label2[j]] += 1
                else:
                    label_distribution[label2[j]] = 0
                
            
            if i == 0:
                print("\nbatch num:", i)
                print("positive_pairs", positive_pairs)
                print("negative_pairs", negative_pairs)
                print("label_distribution", label_distribution)
            # break # debug
        
        print("")
        print("distributions:")
        print("\nbatch num:", i)
        print("positive_pairs", positive_pairs)
        print("negative_pairs", negative_pairs)
        print("label_distribution", label_distribution)
        
        
        
if __name__ == '__main__':
    print("Run this for testing the dataloader only.")

    img_path = "experiments/datasets/2023-02-20_hca_backs"
    preprocessing_path = "experiments/datasets/2023-02-20_hca_backs_preprocessing_opencv"
    seen_classes = ["hca_0", "hca_1", "hca_2", "hca_2a", "hca_3", "hca_4", "hca_5", "hca_6"]
    unseen_classes = ["hca_7", "hca_8", "hca_9", "hca_10", "hca_11", "hca_11a", "hca_12"]
    
    exp_utils.init_seeds(1, cuda_deterministic=False)

    transform_list = [
        # A.SmallestMaxSize(max_size=160),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.1),
        # A.RandomCrop(height=128, width=128),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.1),
        A.RandomBrightnessContrast(p=0.1),
    ]

    # computed transform using compute_mean_std() to give:
    # transform_normalise = transforms.Normalize(mean=[0.5895, 0.5935, 0.6036],
                                # std=[0.1180, 0.1220, 0.1092])
    transform_normalise = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    

    val_transform = A.Compose([
        # transform_normalise,
        # ToTensorV2(),
    ])

    # add normalise
    # transform_list.append(transform_normalise)
    transform_list.append(ToTensorV2())
    train_transform = A.Compose(transform_list)


    dataloader = DataLoaderEvenPairwise(img_path,
                                        preprocessing_path=preprocessing_path,
                                        batch_size=32,
                                        shuffle=True,
                                        seen_classes=seen_classes,
                                        unseen_classes=unseen_classes,
                                        train_transform=train_transform,
                                        val_transform=val_transform)
    
    # dataloader.example(type_name="seen_train")
    dataloader.example(type_name="test")