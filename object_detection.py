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

class ObjectDetection:
    def __init__(self):
        args = types.SimpleNamespace()
        # args.ap_data_file='results/ap_data.pkl'
        # args.bbox_det_file='results/bbox_detections.json'
        # args.benchmark=False
        # args.dataset=None
        # args.detect=False
        # args.emulate_playback=False
        # args.mask_det_file='results/mask_detections.json'
        # args.max_images=-1
        # args.no_bar=False
        # args.no_hash=False
        # args.no_sort=False
        # args.output_coco_json=False
        # args.output_web_json=False
        # args.seed=None
        # args.shuffle=False
        # args.video=None
        # args.video_multiframe=1
        # args.web_det_path='web/dets/'

        args.display=False
        args.display_bboxes=True
        args.display_fps=False
        args.display_lincomb=False
        args.display_masks=True
        args.display_scores=True
        args.display_text=True
        args.trained_model = "yolact/weights/yolact_base_47_60000.pth"
        args.score_threshold = 0.15
        args.top_k = 15
        args.image = "/home/sruiz/datasets/ndds/07-12-2020-front-back-sides-battery/000001.color.png:output.png"
        args.images = None
        args.fast_nms = True
        args.cross_class_nms = False
        args.mask_proto_debug = False
        args.config='yolact_base_config'
        args.crop=True
        args.cuda=True
        eval.args = args

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

    def get_prediction(self, frame):

        with torch.no_grad():
            batch = FastBaseTransform()(frame.unsqueeze(0))
            preds = self.net(batch)

        return preds

    # def post_process(self, preds):
        

    def test(self):
        with torch.no_grad():
            eval.evaluate(self.net, dataset=None) # This works :)

if __name__ == '__main__':
    object_detection = ObjectDetection()
    object_detection.test()

# python yolact/eval.py --trained_model=yolact/weights/yolact_base_39_50000.pth --score_threshold=0.15 --top_k=15 --image=/home/sruiz/datasets/ndds/07-12-2020-front-back-sides-battery/000001.color.png:output.png