import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from scipy.spatial.transform import Rotation
# from utils import timer
import types
from yolact_pkg.layers.output_utils import postprocess, undo_image_transformation
from yolact_pkg.data.config import cfg, set_cfg, set_dataset, MEANS, COLORS
from yolact_pkg.data.coco import COCODetection, get_label_map
from collections import defaultdict

iou_thresholds = [x / 100 for x in range(50, 100, 5)]
coco_cats = {} # Call prep_coco_cats to fill this
coco_cats_inv = {}
color_cache = defaultdict(lambda: {})

def rotated_line(point, angle, length):
    angle_rad = np.deg2rad(angle)
    x2 = point[0] + length * np.cos(angle_rad)
    y2 = point[1] + length * np.sin(angle_rad)
    point2 = tuple([int(x2), int(y2)])

    return point2

def get_labelled_img(img, masks=None, detections=None, h=None, w=None, undo_transform=False, class_color=True, mask_alpha=0.45, fps=None, worksurface_detection=None):

    args = types.SimpleNamespace()
    args.display_masks=True
    args.display_fps=True
    args.display_text=True
    args.display_bboxes=True
    args.display_scores=True

    font_face = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.0
    font_thickness = 1

    if img.shape[0] < 1000:
        font_scale = 0.5

    num_dets_to_consider = len(detections)
    
    info_text = ""
        
    for detection in detections:
        tracking_id = ""
        if not detection.valid:
            tracking_id += "INVALID, "
        tracking_id += "t_id " + str(detection.tracking_id) + ", " if detection.tracking_id is not None else ""
        # tracking_score = "t_score " + str(np.round(detection.tracking_score, 1)) if detection.tracking_score is not None else ""
        info_text +=  detection.label.name + ", " + tracking_id + "\n"

    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """
    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    elif isinstance(img, np.ndarray):
        #! moving to GPU. It might be better to do everything on cpu instead.
        img_gpu = torch.Tensor(img).cuda() / 255.0 
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape

    

    # Quick and dirty lambda for selecting the color for a particular index
    # Also keeps track of a per-gpu color cache for maximum speed
    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (detections[j].label.value * 5 if class_color else j * 5) % len(COLORS)
        
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
            else:
                color = torch.Tensor(color).float() / 255.

            return color

    # TODO: draw invalid detections in light grey with high opacity

    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    if args.display_masks and cfg.eval_mask_branch and num_dets_to_consider > 0 and masks is not None:
        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        if torch.cuda.is_available():
            colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
        else:
            colors = torch.cat([get_color(j).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
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
        fps_text = fps
        if isinstance(fps, float):
            fps_text = str(round(fps, 1)) + " fps"
        text_w, text_h = cv2.getTextSize(fps_text, font_face, font_scale, font_thickness)[0]
        img_gpu[0:text_h+8, 0:text_w+8] *= 0.6 # 1 - Box alpha
    else:
        text_h = 0

    if info_text is not None and info_text != "":
        prev_height = text_h + 8
        for i, line in enumerate(info_text.split('\n')):
            info_text_w, info_text_h = cv2.getTextSize(line, font_face, font_scale, font_thickness)[0]
            img_gpu[prev_height:prev_height+info_text_h+8, 0:info_text_w+8] *= 0.6 # 1 - Box alpha
            prev_height += info_text_h+8

    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255).byte().cpu().numpy()

    # draw work surface corners
    if worksurface_detection is not None:
        for label, coords_in_pixels in worksurface_detection.corners_px_dict.items():
            if coords_in_pixels is not None:
                xc, yc = np.around(coords_in_pixels[:2]).astype(int)
                xc_m, yc_m, *_= worksurface_detection.corners_m_dict[label]

                cv2.circle(img_numpy, (xc, yc), 5, (0, 255, 0), -1)
                cv2.putText(img_numpy, label,
                            (xc - 70, yc-10), font_face, font_scale, [255, 255, 255], font_thickness, cv2.LINE_AA)
                cv2.putText(img_numpy, "(" + str(xc_m) + ", " + str(yc_m) + ")",
                            (xc - 50, yc+30), font_face, font_scale, [255, 255, 255], font_thickness, cv2.LINE_AA)

    for detection in detections:
        outline_colour = (0, 255, 0)
        if not detection.valid:
            outline_colour = (0, 0, 255)

        if detection.center_px is not None:
            cv2.circle(img_numpy, tuple(detection.center_px), 5, outline_colour, -1)
            cv2.drawContours(img_numpy, [detection.obb_px], 0, outline_colour, 2)
            
            # draw the arrow
            point2 = rotated_line(tuple(detection.center_px), detection.angle_px, 60)
            cv2.arrowedLine(img_numpy, tuple(detection.center_px), point2, outline_colour, 3, tipLength = 0.5)
            
            # cv2.drawContours(img_numpy, [detection.mask_contour], 0, outline_colour, 2)

    if args.display_fps and fps is not None:
        # Draw the text on the CPU
        text_pt = (4, text_h + 2)
        text_color = [255, 255, 255]

        cv2.putText(img_numpy, fps_text, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    if info_text is not None and info_text != "":
        text_color = [0, 255, 0]
        for j, line in enumerate(info_text.split('\n')):
            if line != "":
                color = get_color(j).cpu().detach().numpy() *255
                color = [int(i) for i in color]
                
                text_pt = (4, (text_h+8)+(info_text_h+8)*j +info_text_h+2)
                cv2.putText(img_numpy, line, text_pt, font_face, font_scale, color, font_thickness, cv2.LINE_AA)
    
    if num_dets_to_consider == 0:
        return img_numpy

    if args.display_text or args.display_bboxes:
        for j in reversed(range(num_dets_to_consider)):
            detection = detections[j]
            if detection.valid:
                x1 = int(detection.box_px[0, 0])
                y1 = int(detection.box_px[0, 1])
                x2 = int(detection.box_px[1, 0])
                y2 = int(detection.box_px[1, 1])
                
                color = get_color(j).cpu().detach().numpy() *255
                color = [int(i) for i in color]

                if args.display_bboxes:
                    cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 2)

                if args.display_text:
                    z1_m = 0
                    if detection.center is not None:
                        if len(detection.center) == 2:
                            x1_m, y1_m = detection.center
                        else:
                            x1_m, y1_m, z1_m = detection.center
                    else:
                        x1_m, y1_m = (-1, -1)

                    tracking_id = "t_id " + str(detection.tracking_id) + "," if detection.tracking_id is not None else ""
                    # tracking_score = "t_score " + str(np.round(detection.tracking_score, 1)) + ", " if detection.tracking_score is not None else ""
                    text_str = '%s: %s %.2f, (%.2f, %.2f, %.2f)' % (detection.label.name, tracking_id, detection.score, x1_m, y1_m, z1_m) if args.display_scores else detection.label.name

                    text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
                    text_w = int(text_w)
                    text_h = int(text_h)

                    text_pt = (x1, y1 - 3)
                    text_color = (int(255), int(255), int(255))

                    cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                    cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    # show tracking obbs
    for detection in detections:
        if detection.valid and detection.tracking_box is not None:
            x1, y1, x2, y2 = detection.tracking_box
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            color = [190, 77, 37]
            
            # bbox
            #! not showing this right now because it makes visualisation messy
            # cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)
            
            # text_str = 'tracking: %d, %.2f' % (detection.online_id, detection.online_score)

            # text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
            # text_w = int(text_w)
            # text_h = int(text_h)

            # text_pt = (x1, y1 - 3)
            # text_color = (int(255), int(255), int(255))

            # cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
            # cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
        
            
    return img_numpy