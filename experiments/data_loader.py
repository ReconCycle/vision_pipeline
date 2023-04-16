from torchvision import datasets, transforms
from torchvision.utils import make_grid
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time
import PIL.Image as Image
import numpy as np
import cv2
import os
import sys
import regex as re
from tqdm import tqdm
from rich import print
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy


# do as if we are in the parent directory
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from graph_relations import GraphRelations
from object_reid import ObjectReId
from exp_utils import scale_img


class ImageDataset(datasets.ImageFolder):
    def __init__(self,
                 main_path,
                 preprocessing_path=None,
                 class_dirs=[],
                 unseen_class_offset=0,
                 transform=None,
                 exemplar_transform=None,
                 limit_imgs_per_class=None):
        
        self.main_path = main_path
        self.preprocessing_path = preprocessing_path
        
        self.class_dirs = class_dirs
        self.unseen_class_offset = unseen_class_offset
        self.exemplar_transform = exemplar_transform
        
        
        if limit_imgs_per_class is not None:
            print("\n"+"="*20)
            print("Imgs per class limited to", limit_imgs_per_class)
            print("="*20, "\n")
            # to make sure each class is of approximately the same size, we set the number of images per class to 30
            def is_valid_file(file_path):
                file_name = os.path.basename(file_path)
                num = int(re.findall(r'\d+', file_name)[-1])
                
                if num < limit_imgs_per_class:
                    return True
                else:
                    return False
        else:
            is_valid_file = None
        
        
        super(ImageDataset, self).__init__(main_path, transform, is_valid_file=is_valid_file)
        
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, label) where target is class_index of the target class.
        """
        path, label = self.samples[index]

        sample = cv2.imread(path)
        
        # convert to rgb if it is an rgbd image
        # sample = np.array(sample)
        # sample = sample[:, :, :3]
        # sample = Image.fromarray(sample)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
            
        label = label + self.unseen_class_offset

        dirname = os.path.basename(os.path.dirname(path))
        
        # get preprocessed detections for img (if they exist)
        detections = []
        if self.preprocessing_path is not None:
            
            filename = os.path.basename(path)
            
            file_path = os.path.join(self.preprocessing_path, dirname, filename + ".json")
            if os.path.isfile(file_path):

                try:
                    with open(file_path, 'r') as json_file:
                        detections = jsonpickle.decode(json_file.read(), keys=True)
                        
                except ValueError as e:
                    print("couldn't read json file properly: ", e)
            else:
                print("[red]detection file doesn't exist:" + file_path + "[/red]")
        
            graph = GraphRelations(detections)

            # form groups, adds group_id property to detections
            graph.make_groups()
            
            sample, poly = ObjectReId.find_and_crop_det(sample, graph)
            
            # apply albumentations transform
            sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
            sample = self.transform(image=sample)["image"]
        
        else:
            # for the preprocessing step
            sample = np.array(sample)
        
        # if needed we could pass detections and original image too
        return sample, label, path, poly
        
    # restrict classes to those in subfolder_dirs
    def find_classes(self, dir: str):
        classes = [d.name for d in os.scandir(dir) if d.is_dir() and d.name in self.class_dirs]
        classes.sort()
        # print("Directories in this dataset:", classes)
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
    
    
class DataLoader():
    def __init__(self,
                 img_path,
                 preprocessing_path=None,
                 batch_size=16,
                 num_workers=8,
                 validation_split=.2,
                 shuffle=True,
                 seen_classes=[],
                 unseen_classes=[],
                 train_transform=None,
                 val_transform=None,
                 limit_imgs_per_class=None,
                 cuda=True):
        
        random_seed= 42
        
        self.img_path = img_path
        self.batch_size = batch_size
        
        # use seen_classes and unseen_classes to specify which directories to load
        self.seen_classes = seen_classes
        self.unseen_classes = unseen_classes

        self.seen_dirs = seen_classes
        self.unseen_dirs = unseen_classes



        # train_tf_dataset = datasets.StanfordCars(root="/home/sruiz/datasets2", 
        #                                         split="train", 
        #                                         download=True, 
        #                                         transform=train_transform)
        
        # val_tf_dataset = datasets.StanfordCars(root="/home/sruiz/datasets2", 
        #                                         split="test",
        #                                         download=True,
        #                                         transform=val_transform)
                
        #! note the only difference is the transform!
        train_tf_dataset = ImageDataset(img_path,
                                    preprocessing_path,
                                    self.seen_dirs,
                                    transform=train_transform,
                                    limit_imgs_per_class=limit_imgs_per_class)
    
        val_tf_dataset = ImageDataset(img_path,
                                    preprocessing_path,
                                    self.seen_dirs,
                                    transform=val_transform,
                                    limit_imgs_per_class=limit_imgs_per_class)
        
        len_dataset = len(train_tf_dataset)

        # create seen train/val/test split
        generator = torch.Generator().manual_seed(random_seed)
        len_seen_train = int((1.0 - 2*validation_split) * len_dataset) # 0.6/0.2/0.2 split
        len_seen_val = int(validation_split * len_dataset)
        len_seen_test = len_dataset - len_seen_train - len_seen_val

        seen_train_idxs, seen_val_idxs, seen_test_idxs = torch.utils.data.random_split(
            np.arange(len_dataset),
            (len_seen_train, len_seen_val, len_seen_test),
            generator=generator
        )

        self.datasets = {}
        self.datasets["seen_train"] = torch.utils.data.Subset(train_tf_dataset, seen_train_idxs)
        self.datasets["seen_val"] = torch.utils.data.Subset(val_tf_dataset, seen_val_idxs)
        self.datasets["seen_test"] = torch.utils.data.Subset(val_tf_dataset, seen_test_idxs)
        
        # add unseen dataset
        self.datasets["unseen_test"] = ImageDataset(img_path,
                                                preprocessing_path,
                                                self.unseen_dirs,
                                                unseen_class_offset=len(train_tf_dataset.classes),
                                                transform=val_transform,
                                                limit_imgs_per_class=limit_imgs_per_class)
        
        # concat seen_test and unseen_test datasets
        self.datasets["test"] = torch.utils.data.ConcatDataset([self.datasets["seen_test"], self.datasets["unseen_test"]])
        
        # create the dataloaders
        # todo: fix bug, either requiring: generator=torch.Generator(device='cuda'),
        # todo: or requiring shuffle=False
        generator = None
        if cuda:
            generator = torch.Generator(device='cuda')
            
        
        def custom_collate(instances):
            # handle sample, label, and path like normal
            # but handle detections as list of lists.
            
            # elem = instances[0] # tuple: (sample, label, path, detections)
            
            batch = []
            
            for i in range(len(instances[0])):
                batch.append([instance[i] for instance in instances])

            # print("batch[0]", len(batch[0]), type(batch[0][0]))
            
            # apply default collate for: sample, label, path
            batch[0] = torch.utils.data.default_collate(batch[0])
            batch[1] = torch.utils.data.default_collate(batch[1])
            batch[2] = torch.utils.data.default_collate(batch[2])

            return batch
        
        
        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x],
                                                           num_workers=num_workers,
                                                           batch_size=batch_size,
                                                           generator=generator,
                                                           shuffle=shuffle,
                                                           collate_fn=custom_collate)
                            for x in ["seen_train", "seen_val", "seen_test", "unseen_test", "test"]}
        
        self.dataset_lens = {x: len(self.datasets[x]) for x in ["seen_train", "seen_val", "seen_test", "unseen_test", "test"]}
        
        self.classes = {
            "seen_train": train_tf_dataset.classes,
            "seen_val": train_tf_dataset.classes,
            "seen_test": train_tf_dataset.classes,
            "unseen_test": self.datasets["unseen_test"].classes,
            "test": np.concatenate((train_tf_dataset.classes, self.datasets["unseen_test"].classes)),
            "all": np.concatenate((train_tf_dataset.classes, self.datasets["unseen_test"].classes))
        }
        
        
    def compute_mean_std(self):
        # compute mean and std for training dataset of images. From here the normalisation values can be set.
        # for example:
        # A.Normalize(mean=(154.5703, 148.4985, 152.1174), std=(31.3750, 29.3120, 29.2421), max_pixel_value=1),
        # after setting this, the (mean, std) should be (0, 1)
        
        batches = []
        for sample, label, *_ in tqdm(self.dataloaders["seen_train"]):
            batches.append(sample)
            # print("sample", type(sample), len(sample), type(sample[0]))
        
        all_imgs = torch.cat(batches)
        all_imgs = torch.Tensor.float(all_imgs)
        
        print("all_imgs.shape", all_imgs.shape)
        index_channels = all_imgs.shape.index(3)
        if index_channels != 3:
            # channels are not in the last dimension
            # move the channels to the last dimension:
            all_imgs = torch.transpose(all_imgs, index_channels, -1).contiguous()
        
        if all_imgs.shape[-1] != 3:
            print("\nERROR: the last dimension should be the channels!\n")
        
        print("all_imgs.shape", all_imgs.shape)
        
        all_imgs = all_imgs.view(-1, 3)
        
        print("all_imgs.shape", all_imgs.shape)
        
        min = all_imgs.min(dim=0).values
        max = all_imgs.max(dim=0).values
        
        print("min", min)
        print("max", max)
        
        mean = all_imgs.mean(dim=0)
        std = all_imgs.std(dim=0)
        print("mean", mean)
        print("std", std)
    
    def example_iterate(self, type_name):
        label_distribution = {}
        for a_class in self.classes[type_name]:
            label_distribution[a_class] = 0
        
        for i, (sample, labels, path, detections) in enumerate(self.dataloaders[type_name]):
            # print("i", i)
            # print("inputs.shape", inputs.shape)
            # print("labels", labels)
            # print("")
            # sample[0]

            print(f"len(labels) {len(labels)}")

            img_grid = make_grid(sample, nrow=4)
            img_grid = transforms.ToPILImage()(img_grid)
            # img_grid.show()
            img_grid = np.array(img_grid)
            img_grid = cv2.cvtColor(img_grid, cv2.COLOR_RGB2BGR)
            cv2.imshow("img_grid", scale_img(img_grid))
            k = cv2.waitKey(0)

            # labels_name = [self.classes[type_name][label] for label in labels]

            # for j in np.arange(len(labels)):
            #     label_distribution[labels_name[j]] += 1

        return label_distribution

    def example(self):
        
        print("classes:", self.classes)
        print("")

        label_dist_seen_train = dataloader.example_iterate("seen_train")
        label_dist_seen_val = dataloader.example_iterate("seen_val")

        print("label_dist_seen_train", label_dist_seen_train)
        print("label_dist_seen_val", label_dist_seen_val)
        
        
if __name__ == '__main__':
    print("Run this for testing the dataloader only.")
    # img_path = "/home/sruiz/datasets2/reconcycle/simon_rgbd_dataset/hca_simon/sorted_in_folders"
    # img_path = "experiments/datasets/hca_simon/sorted_in_folders"
    img_path = "experiments/datasets/2023-02-20_hca_backs"
    preprocessing_path = "experiments/datasets/2023-02-20_hca_backs_preprocessing_opencv"
    seen_classes = ["hca_0", "hca_1", "hca_2", "hca_2a", "hca_3", "hca_4", "hca_5", "hca_6"]
    unseen_classes = ["hca_7", "hca_8", "hca_9", "hca_10", "hca_11", "hca_11a", "hca_12"]

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
        ToTensorV2(),
    ])

    # add normalise
    # transform_list.append(transform_normalise)
    transform_list.append(ToTensorV2())
    train_transform = A.Compose(transform_list)

    dataloader = DataLoader(img_path,
                            preprocessing_path=preprocessing_path,
                            seen_classes=seen_classes,
                            unseen_classes=unseen_classes,
                            train_transform=train_transform,
                            val_transform=val_transform,
                            cuda=False)
    
    # dataloader.example_iterate(type_name="seen_train")
    dataloader.example_iterate(type_name="test")
