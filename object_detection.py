import sys, os
# having trouble importing the yolact directory. Doing this as a workaround:
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolact'))
from yolact import Yolact
import eval
import torch
import torch.backends.cudnn as cudnn
import types
import cv2
from utils.augmentations import BaseTransform, FastBaseTransform, Resize

# post process function
from utils import timer
from layers.output_utils import postprocess, undo_image_transformation
from data import cfg
import obb
import numpy as np

class ObjectDetection:
    def __init__(self, trained_model, config_name='coco_ndds', score_threshold=0.15):
        args = types.SimpleNamespace()

        args.display=False
        args.display_lincomb=False
        args.trained_model = trained_model
        args.score_threshold = score_threshold
        args.top_k = 15
        args.mask_proto_debug = False
        args.config= config_name
        args.crop=True
        args.cuda=True
        eval.args = args
        self.args = args

        self.h = None
        self.w = None

        self.check_gpu()

        with torch.no_grad():
            if not os.path.exists('results'):
                os.makedirs('results')

            if args.cuda:
                cudnn.fastest = True
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
            else:
                torch.set_default_tensor_type('torch.FloatTensor')

            print('Loading model...', end='')
            net = Yolact()
            net.load_weights(args.trained_model)
            net.eval()
            print(' Done.')

            if args.cuda:
                net = net.cuda()

            self.net = net

    def check_gpu(self):
        cuda_device = torch.cuda.current_device()
        print("current cuda device:", cuda_device)
        print("device name:", torch.cuda.get_device_name(cuda_device))
        print("device count:", torch.cuda.device_count())
        print("cuda available:", torch.cuda.is_available())

    def get_prediction(self, frame):

        self.h, self.w, _ = frame.shape
        with torch.no_grad():
            batch = FastBaseTransform()(frame.unsqueeze(0))
            preds = self.net(batch)

        return preds

    def post_process(self, preds, w=None, h=None):

        with timer.env('Postprocess'):
            cfg.mask_proto_debug = self.args.mask_proto_debug

            if w is None and h is None:
                w = self.w
                h = self.h

            t = postprocess(preds, w, h, visualize_lincomb = self.args.display_lincomb,
                                            crop_masks        = self.args.crop,
                                            score_threshold   = self.args.score_threshold)
            # cfg.rescore_bbox = save

        with timer.env('Copy'):
            idx = t[1].argsort(0, descending=True)[:self.args.top_k]
            
            if cfg.eval_mask_branch:
                # Masks are drawn on the GPU, so don't copy
                masks = t[3][idx]
            classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

        num_dets_to_consider = min(self.args.top_k, classes.shape[0])
        for j in range(num_dets_to_consider):
            if scores[j] < self.args.score_threshold:
                num_dets_to_consider = j
                break

        if cfg.eval_mask_branch and num_dets_to_consider > 0:
            # After this, mask is of size [num_dets, h, w, 1]
            masks = masks[:num_dets_to_consider, :, :, None]

        # calculate the oriented bounding boxes
        obb_corners = []
        obb_centers = []
        for i in np.arange(len(masks)):
            obb_mask = masks[i].cpu().numpy()[:,:, 0] == 1
            corners, center = obb.get_obb_from_mask(obb_mask)
            obb_corners.append(corners)
            obb_centers.append(center)

        return classes, scores, boxes, masks, obb_corners, obb_centers, num_dets_to_consider

    def test(self):
        with torch.no_grad():
            eval.evaluate(self.net, dataset=None) # This works :)

if __name__ == '__main__':
    object_detection = ObjectDetection(trained_model="yolact/weights/yolact_base_47_60000.pth")
    object_detection.test()

# python yolact/eval.py --trained_model=yolact/weights/yolact_base_39_50000.pth --score_threshold=0.15 --top_k=15 --image=/home/sruiz/datasets/ndds/07-12-2020-front-back-sides-battery/000001.color.png:output.png