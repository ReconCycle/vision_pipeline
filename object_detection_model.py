import sys
import os
import cv2
import numpy as np
import json
import argparse
import time
import atexit
from datetime import datetime
from enum import Enum
from rich import print
import commentjson
from types import SimpleNamespace
import torch
from tqdm import tqdm

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from yolact_pkg.data.config import Config
from yolact_pkg.yolact import Yolact

from ultralytics import YOLO

from device_reid.model_classify import ClassifyModel
from vision_pipeline.object_reid_superglue import ObjectReIdSuperGlue
import imutils


class ModelType(Enum):
    yolact = 1
    yolov8 = 2


class ObjectDetectionModel:
    def __init__(self, config) -> None:
        # model that can load yolov8 or yolact

        self.config = config
        self.model_type = None
        self.yolact = None
        self.yolov8 = None
        self.dataset = None

        self.classify_model = None
        self.superglue_model = None
        self.superglue_templates = {}

        if config.model.lower() == "yolact":
            self.model_type = ModelType.yolact
            self.yolact, self.dataset = self.load_yolact(config)

        elif config.model.lower() == "yolov8":
            self.model_type = ModelType.yolov8
            self.yolov8, self.dataset = self.load_yolov8(config)

        self.load_classify_model()
        self.load_superglue_model()
        self.load_superglue_templates(config.superglue_templates) # todo: add to config


    def load_yolact(self, yolact_config):
        yolact_dataset = None
        
        if os.path.isfile(yolact_config.yolact_dataset_file):
            print("loading", yolact_config.yolact_dataset_file)
            with open(yolact_config.yolact_dataset_file, "r") as read_file:
                yolact_dataset = commentjson.load(read_file)
                print("yolact_dataset", yolact_dataset)
        else:
            raise Exception("config.yolact_dataset_file is incorrect: " +  str(yolact_config.yolact_dataset_file))
                
        model_path = None
        if "model" in yolact_dataset:
            model_path = os.path.join(os.path.dirname(yolact_config.yolact_dataset_file), yolact_dataset["model"])

        yolact_dataset = Config(yolact_dataset)
        
        config_override = {
            'name': 'yolact_base',

            # Dataset stuff
            'dataset': yolact_dataset,
            'num_classes': len(yolact_dataset.class_names) + 1,

            # Image Size
            'max_size': 1100,

            # These are in BGR and are for ImageNet
            'MEANS': (103.94, 116.78, 123.68),
            'STD': (57.38, 57.12, 58.40),
            
            # the save path should contain resnet101_reducedfc.pth
            'save_path': './data_limited/yolact/',
            'score_threshold': yolact_config.yolact_score_threshold,
            'top_k': len(yolact_dataset.class_names)
        }
            
        print("model_path", model_path)
        
        yolact = Yolact(config_override)
        yolact.cfg.print()
        yolact.eval()
        yolact.load_weights(model_path)
        
        return yolact, yolact_dataset


    def load_yolov8(self, config):
        # Load finetuned YOLOv8m-seg model
        yolov8 = YOLO(os.path.expanduser(config.yolov8_model_file))
        class_names_dict = yolov8.names

        yolov8_dataset = SimpleNamespace()
        yolov8_dataset.class_names = list(class_names_dict.values())

        return yolov8, yolov8_dataset


    def infer_yolov8(self, colour_img):

        # Run batched inference on a list of images
        results = self.yolov8.predict(colour_img, 
              save=False,
              retina_masks=True,
              imgsz=640,
              verbose=False,
              conf=self.config.yolov8_score_threshold)
        
        if len(results) >= 1:
            result = results[0]
        
            data = result.boxes  # Boxes object for bbox outputs

            scores = data.conf.cpu().numpy()
            classes = data.cls
            classes_int = [int(t.item()) for t in classes]
            boxes = data.xyxy.cpu().numpy() 
            
            masks_results = result.masks
            if masks_results is not None:
                masks = masks_results.data  # Masks object for segmentation masks outputs
                masks = masks.unsqueeze(-1) # make it like yolact
            else:
                masks = torch.empty((0,))

            # print("orig_img", orig_img.shape, type(orig_img))            
            # print("classes", classes.shape) # torch.Size([5])
            # print("scores", scores.shape) # torch.Size([5])
            # print("boxes", boxes.shape) # torch.Size([5, 4])
            # print("masks", masks.shape, type(masks)) # torch.Size([5, 1450, 1450])
            # print("classes", classes)
            # print("classes_int", classes_int)
            # print("scores", scores)

            return colour_img, classes_int, scores, boxes, masks
            
        else:
            return colour_img, np.array([]), np.array([]), np.array([]), torch.empty((0,))

    def infer(self, colour_img):

        if self.model_type == ModelType.yolact:
            frame, classes, scores, boxes, masks = self.yolact.infer(colour_img)

            # all tensors
            # print("frame", frame.shape, type(frame)) # torch.Size([1450, 1450, 3])
            # print("classes", classes.shape, type(classes)) # (4,)
            # print("scores", scores.shape, type(scores)) # (4,)
            # print("boxes", boxes.shape, type(boxes)) # (4, 4)
            # print("masks", masks.shape, type(masks)) # torch.Size([4, 1450, 1450, 1])

            # print("classes", classes) # eg. [ 2  1 10  3]
            # print("scores", scores) # eg. [    0.99976     0.99868     0.99818     0.96942]
            
        elif self.model_type == ModelType.yolov8:
            frame, classes, scores, boxes, masks = self.infer_yolov8(colour_img)

        return frame, classes, scores, boxes, masks


    def load_classify_model(self):
        model_path = os.path.expanduser(self.config.classifier_model_file)
        self.classify_model = ClassifyModel.load_from_checkpoint(model_path, strict=False)

        print("model.learning_rate", self.classify_model.learning_rate)
        print("model.batch_size", self.classify_model.batch_size)
        print("model.freeze_backbone", self.classify_model.freeze_backbone)

    
    def infer_classify(self, sample_cropped):
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]

        transform = A.Compose([
            A.Resize(300, 300),
            A.Normalize(mean=norm_mean, std=norm_std),
            ToTensorV2()
        ])
        if len(sample_cropped.shape) > 3:
            # input is a batch
            img_tensor = torch.stack([transform(image=img)["image"] for img in sample_cropped], dim=0)
            img_tensor = img_tensor.cuda()
        else:
            # input is single image
            img_tensor = transform(image=sample_cropped)["image"]
            img_tensor = img_tensor.unsqueeze(0).cuda() # batch size 1

        self.classify_model.eval()

        with torch.no_grad():
            logits = self.classify_model(img_tensor)

            preds = torch.argmax(logits, dim=1)

            # print("logits.shape", logits.shape)
            # print("preds.shape", preds.shape, preds)

            # for a single image:
            # pred_label = self.classify_model.labels[preds[0]]
            # we took logsoftmax, so we have to take the exp to get confidence
            # conf = torch.exp(logits[0][preds[0]])

            # for a batch:
            pred_label = [self.classify_model.labels[i] for i in preds]

            # we took logsoftmax, so we have to take the exp to get confidence
            # ! there must be a better way to write this
            conf = [float(torch.exp(logits[i][preds[i]]).cpu()) for i in np.arange(len(preds))]
        
        return pred_label, conf
    
    def load_superglue_templates(self, template_dir):

        template_dir = os.path.expanduser(template_dir)

        if not os.path.isdir(template_dir):
            print("[red]template_dir not found!", template_dir)

        subfolders = [ (f.path, f.name) for f in os.scandir(template_dir) if f.is_dir() and len(os.listdir(f)) > 0]
        for sub_path, sub_name in tqdm(subfolders):
            files = [f for f in os.listdir(sub_path) if 
                        os.path.isfile(os.path.join(sub_path, f)) 
                        and os.path.join(sub_path, f).endswith(('.png', '.jpg', '.jpeg'))
                        and "template" in f]
                
            if len(files) > 0:

                # get the first image and use it as template
                file = files[0]

                img = cv2.imread(os.path.join(sub_path, file))
                self.superglue_templates[sub_name] = img
            else:
                print(f"[red]can't find template for: {sub_name}")

        print(f"[green]Loaded superglue templates {len(self.superglue_templates)}")


    def load_superglue_model(self):
        model_path = os.path.expanduser(self.config.superglue_model_file)
        self.superglue_model = ObjectReIdSuperGlue(model_path, self.config.superglue_match_threshold)

    def superglue_rot_estimation(self, sample, label, visualise=None, save_visualise=None):

        # get the template image from the loaded templates
        img1 = self.superglue_templates[label]
        # print("superglue_rot_estimation: img1.shape", img1.shape)
        img1 = imutils.resize(img1, width=400, height=400) #! the input sample must be the same size as this.
        # print("superglue_rot_estimation: img1.shape", img1.shape)
        if sample.shape != img1.shape:
            raise ValueError(f"shapes for superglue input do not match: {img1.shape}, {sample.shape}")

        # display(PILImage.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)))

        img2 = sample

        if visualise is None:
            visualise = self.config.superglue_visualise_to_file
            
        if save_visualise is None:
            save_visualise = self.config.superglue_visualise_to_file

        # img1 = exp_utils.torch_to_grayscale_np_img(img1).astype(np.float32)
        # img2 = exp_utils.torch_to_grayscale_np_img(img2).astype(np.float32)
        img1_grey = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
        img2_grey = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)

        affine_score, score_ratio, mconf, median_affine_error, len_matches, vis_out, angle, est_homo_ransac, angle_from_similarity, angle_from_est_affine_partial = self.superglue_model.compare(img1_grey, img2_grey, gt=True, affine_fit=False, visualise=visualise, debug=self.config.debug)


        # if visualise:
        #     sns.histplot(mconf)
        #     plt.show()

        # if visualise and vis_out is not None:            
        #     display(PILImage.fromarray(vis_out))

        if save_visualise:       
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            if not os.path.exists("saves/"):
                os.makedirs("saves/")
            filename = f"saves/{timestamp}_{label}_matches_{len_matches}.jpg"
            
            print("\n[red]************************************")
            print(f"[red]saving superglue vis: {filename}")
            print("[red]************************************\n")   
            
            cv2.imwrite(filename, vis_out)

        
        return angle, vis_out, est_homo_ransac, angle_from_similarity, angle_from_est_affine_partial