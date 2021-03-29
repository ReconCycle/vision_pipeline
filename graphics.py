import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
# from utils import timer
import types
from yolact.layers.output_utils import postprocess, undo_image_transformation
from yolact.data import cfg, set_cfg, set_dataset
from yolact.data import COCODetection, get_label_map, MEANS, COLORS
from collections import defaultdict

iou_thresholds = [x / 100 for x in range(50, 100, 5)]
coco_cats = {} # Call prep_coco_cats to fill this
coco_cats_inv = {}
color_cache = defaultdict(lambda: {})

def get_labelled_img(img, classes, scores, boxes, masks, obb_corners, obb_centers, num_dets_to_consider, h=None, w=None, undo_transform=False, class_color=True, mask_alpha=0.45, fps=None, worksurface_detection=None):

    args = types.SimpleNamespace()
    args.display_masks=True
    args.display_fps=True
    args.display_text=True
    args.display_bboxes=True
    args.display_scores=True

    font_face = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.6
    font_thickness = 1

    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """
    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape

    # Quick and dirty lambda for selecting the color for a particular index
    # Also keeps track of a per-gpu color cache for maximum speed
    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)
        
        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            if not undo_transform:
                # The image might come in as RGB or BRG, depending
                color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
            return color

    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    if args.display_masks and cfg.eval_mask_branch and num_dets_to_consider > 0:
        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = masks * (-mask_alpha) + 1

        # I did the math for this on pen and paper. This whole block should be equivalent to:
        #    for j in range(num_dets_to_consider):
        #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)

        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand
    
    if args.display_fps and fps is not None:
        # Draw the box for the fps on the GPU
        fps_text = str(round(fps, 1)) + " fps"
        text_w, text_h = cv2.getTextSize(fps_text, font_face, font_scale, font_thickness)[0]
        img_gpu[0:text_h+8, 0:text_w+8] *= 0.6 # 1 - Box alpha


    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255).byte().cpu().numpy()

    # draw work surface corners
    if worksurface_detection is not None:
        for i in range(len(worksurface_detection.corners_in_pixels)):
            xc, yc = worksurface_detection.corners_in_pixels[i]
            xc_meters, yc_m = worksurface_detection.corners_in_meters[i]
            print("xc, yc", xc, yc)
            cv2.circle(img_numpy, (xc, yc), 5, (0, 255, 0), -1)
            print(worksurface_detection.corner_labels[i] + ", (" + str(xc_meters) + ", " + str(yc_m) + ")")
            cv2.putText(img_numpy, worksurface_detection.corner_labels[i] + ", (" + str(xc_meters) + ", " + str(yc_m) + ")",
                        (xc, yc), font_face, font_scale, [255, 255, 255], font_thickness, cv2.LINE_AA)


    # draw oriented bounding boxes
    for i in np.arange(len(obb_centers)):
        cv2.circle(img_numpy, tuple(obb_centers[i]), 5, (0, 255, 0), -1)
        for j in np.arange(4):
            cv2.line(img_numpy, tuple(obb_corners[i][j]), tuple(obb_corners[i][j+1]), (0, 255, 0), thickness=2)

    if args.display_fps and fps is not None:
        # Draw the text on the CPU
        text_pt = (4, text_h + 2)
        text_color = [255, 255, 255]

        cv2.putText(img_numpy, fps_text, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    if num_dets_to_consider == 0:
        return img_numpy

    if args.display_text or args.display_bboxes:
        for j in reversed(range(num_dets_to_consider)):
            x1, y1, x2, y2 = boxes[j, :]
            x1_center, y1_center = obb_centers[j]
            color = get_color(j)
            score = scores[j]

            if args.display_bboxes:
                cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

            if args.display_text:
                _class = cfg.dataset.class_names[classes[j]]

                x1_m, y1_m = worksurface_detection.pixels_to_meters((x1_center, y1_center)).tolist()

                text_str = '%s: %.2f, (%.2f, %.2f)' % (_class, score, x1_m, y1_m) if args.display_scores else _class
                print(text_str)

                text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                text_pt = (x1, y1 - 3)
                text_color = [255, 255, 255]

                cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
            
    return img_numpy