import numpy as np
import cv2
from types import SimpleNamespace
import enum
import time

from yolact_pkg.data.config import Config
from yolact_pkg.yolact import Yolact
from yolact_pkg.eval import annotate_img
from yolact_pkg.data.config import resnet101_rgbd_backbone 

from tracker.byte_tracker import BYTETracker

from helpers import Struct, Detection
import graphics

class NNGapDetector:
    def __init__(self):

        self.dataset = Config({
            'name': 'Base Dataset',
            # Training folder 
            'train_images': '/Users/simonblaue/Desktop/YOLACT_RGBD/datasets/heat allocators/train/',
            # Train annotaions json file
            'train_info':   '/Users/simonblaue/Desktop/YOLACT_RGBD/datasets/heat allocators/train/annotations.json',
            # Validation folder
            'valid_images': '/Users/simonblaue/Desktop/YOLACT_RGBD/datasets/heat allocators/val/',
            # Validation annotaions json file
            'valid_info':   '/Users/simonblaue/Desktop/YOLACT_RGBD/datasets/heat allocators/val/annotations.json',
            # Whether or not to load GT. If this is False, eval.py quantitative evaluation won't work.
            'has_gt': True,
            # A list of names for each of you classes.
            'class_names': ('gap'),
            # COCO class ids aren't sequential, so this is a bandage fix. If your ids aren't sequential,
            # provide a map from category_id -> index in class_names + 1 (the +1 is there because it's 1-indexed).
            # If not specified, this just assumes category ids start at 1 and increase sequentially.
            'label_map': None
        })
            
        config_override = {
            'name': 'yolact_base',
            # Dataset stuff
            'dataset': self.dataset,
            'num_classes': len(self.dataset.class_names) +1,
            # Image Size
            'max_size': 512,
            
            'save_path': 'data_limited/yolact_rgbd/',
            
            # we can override args used in eval.py:        
            'score_threshold': 0.1,
            'top_k': 10,
            
            # Change these for different nets
            'MEANS': (116.24457136111748,119.55194544312776,117.05760736644808,196.36951043344453), 
            'STD': (1.4380884974626822,1.8110670756137501,1.5662493838264602,1.8686978397590024),
            'augment_photometric_distort': False,
            'backbone': resnet101_rgbd_backbone.copy({  # change backbone for different models
                'selected_layers': list(range(1, 4)),
                'use_pixel_scales': True,
                'preapply_sqrt': False,
                'use_square_anchors': True, # This is for backward compatability with a bug
                'pred_aspect_ratios': [ [[1, 1/2, 2]] ]*5,
                'pred_scales': [[24], [48], [96], [192], [384]],
            }),
        }

        # we can override training args here:
        training_args_override = {
            "batch_size": 8,
            "save_interval": -1, # -1 for saving only at end of the epoch
        }

        # we can override inference args here
        self.override_eval_config = Config({
                'cuda' : False,
                'top_k': 10,
                'score_threshold': 0.1,
                'display_masks': True,
                'display_fps' : False,
                'display_text': True,
                'display_bboxes': False,
                'display_scores': True,
                'save_path': 'data_limited/yolact_rgbd/',
                'MEANS': (116.24457136111748,119.55194544312776,117.05760736644808,196.36951043344453),
                'STD': (1.4380884974626822,1.8110670756137501,1.5662493838264602,1.8686978397590024),
                
        })

        model_path = "data_limited/yolact_rgbd/2022-08-03/yolact_base_118_1309.pth"

        self.yolact = Yolact(config_override)
        self.yolact.cfg.print()
        self.yolact.eval()
        self.yolact.load_weights(model_path)

        self.tracker_args = SimpleNamespace()
        self.tracker_args.track_thresh = 0.1
        self.tracker_args.track_buffer = 10 # num of frames to remember lost tracks
        self.tracker_args.match_thresh = 2.5 # default: 0.9 # higher number is more lenient
        self.tracker_args.min_box_area = 10
        self.tracker_args.mot20 = False
        
        self.tracker = BYTETracker(self.tracker_args)
        self.fps_graphics = -1.
        self.fps_objdet = -1.

        # convert class names to enums
        labels = enum.IntEnum('label', self.dataset.class_names, start=0)
        self.labels = labels

        print("labels", labels)

    @staticmethod
    def nan_in_list(a):
        return True if True in np.isnan(np.array(a)) else False

    def get_prediction(self, colour_img, depth_img, extra_text=None):

        print("colour_img.shape", colour_img.shape)
        print("depth_img.shape", depth_img.shape)

        img = np.dstack((colour_img,depth_img)).astype(np.int16)

        print("img.shape", img.shape)

        t_start = time.time()
        
        frame, classes, scores, boxes, masks = self.yolact.infer(img)
        print("classes", classes)

        print("frame.shape", frame.shape)
        fps_nn = 1.0 / (time.time() - t_start)

        detections = []
        # new_scores = []
        # new_boxes = []
        # new_masks = []
        # new_classes = []

        well_defined_mask = []
        for i in np.arange(len(classes)):
            
            # this could maybe be implemented inside the yolact pkg
            if boxes[i][0] >= boxes[i][2] or boxes[i][1] >= boxes[i][3]:
                print("box not well defined")
                well_defined_mask.append(False)
            else:
                well_defined_mask.append(True)
                # new_scores.append(scores[i])
                # new_boxes.append(boxes[i])
                # new_masks.append(masks[i])
                # new_classes.append(classes[i])

                detection = Detection()
                detection.id = i
                
                # ! -1 here because Simon's network is broken somehow
                detection.label = self.labels(classes[i] - 1) # self.dataset.class_names[classes[i]]
                
                detection.score = float(scores[i])
                detection.box = boxes[i]
                detection.mask = masks[i]
                
                # compute contour. Required for obb and graph_relations
                mask = masks[i].cpu().numpy().astype("uint8")
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(cnts) > 0:
                    detection.mask_contour = np.squeeze(cnts[0])
                
                detections.append(detection)
        # scores = np.array(new_scores)
        # boxes = np.array(new_boxes)
        # masks = np.array(new_masks)
        # classes = np.array(new_classes)
        classes = classes[well_defined_mask]
        scores = scores[well_defined_mask]
        boxes = boxes[well_defined_mask]
        masks = masks[well_defined_mask]
        



        tracker_start = time.time()
        # apply tracker
        # look at: https://github.com/ifzhang/ByteTrack/blob/main/yolox/evaluators/mot_evaluator.py
        online_targets = self.tracker.update(boxes, scores) #? does the tracker not benefit from the predicted classes?

        for t in online_targets:
            detections[t.input_id].tracking_id = t.track_id
            detections[t.input_id].tracking_box = t.tlbr
            detections[t.input_id].score = float(t.score)

        fps_tracker = 1.0 / (time.time() - tracker_start)

        
        graphics_start = time.time()
        if extra_text is not None:
            extra_text + ", "
        else:
            extra_text = ""
        fps_str = extra_text + "objdet: " + str(round(self.fps_objdet, 1)) + ", nn: " + str(round(fps_nn, 1)) + ", tracker: " + str(np.int(round(fps_tracker, 0))) + ", graphics: " + str(np.int(round(self.fps_graphics, 0)))
        
        frame_rgb = frame[:, :, :3]
        print("frame_rgb.shape", frame_rgb.shape)
        labelled_img = graphics.get_labelled_img(frame_rgb, masks, detections, fps=fps_str, worksurface_detection=None)

        # labelled_img = annotate_img(frame, classes, scores, boxes, masks, override_args=self.override_eval_config)

        print("labelled_img.shape", labelled_img.shape, type(labelled_img))

        # labelled_img_rgb = labelled_img[:, :, :]

        # print("labelled_img_rgb.shape", labelled_img_rgb.shape)

        self.fps_graphics = 1.0 / (time.time() - graphics_start)
        self.fps_objdet = 1.0 / (time.time() - t_start)

        return labelled_img

