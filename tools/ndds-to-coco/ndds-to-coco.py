import datetime
import json
import os
import numpy as np
import pycococreatortools
import regex
from helpers import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import time


DATA_DIR = '/home/sruiz/datasets/ndds/13-01-2021-segmented-battery'
JSON_PATH = os.path.join(DATA_DIR, "_coco.json")

################################################################################
## !FIRST USE TESTING_STAGE = True
################################################################################
TESTING_STAGE = False

INFO = {
    "description": "NDDS Dataset",
    "url": "https://github.com/sebastian-ruiz",
    "version": "0.1.0",
    "year": 2021,
    "contributor": "Sebastian Ruiz",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

# These should be put in the correct order starting with id:0, id:1, etc...
# If we set START_AT_0 = False then we want the order to be id:1, id:2, ...
START_AT_0 = False
CATEGORIES = [
    {
        'id': 1,
        'name': 'background',
        'supercategory': 'background',
    },
    {
        'id': 2,
        'name': 'back',
        'supercategory': 'hca',
    },
    {
        'id': 3,
        'name': 'battery',
        'supercategory': 'hca',
    },
    {
        'id': 4,
        'name': 'front',
        'supercategory': 'hca',
    },
    {
        'id': 5,
        'name': 'internals',
        'supercategory': 'hca',
    },
    {
        'id': 6,
        'name': 'pcb',
        'supercategory': 'hca',
    },
    {
        'id': 7,
        'name': 'side2',
        'supercategory': 'hca',
    },
    {
        'id': 8,
        'name': 'side1',
        'supercategory': 'hca',
    },
]


def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

def main():
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    total_is_id = 0

    data_dir = os.path.join(DATA_DIR)

    images_fps = [img for img in listdir_nohidden(data_dir) if img.endswith(".color.png")]
    images_fps.sort(key=lambda f: int(regex.sub('\D', '', f)))
    images_fps = np.array([os.path.join(data_dir, image) for image in images_fps])

    masks_label_fps = [img for img in listdir_nohidden(data_dir) if img.endswith(".label.png")]  # class segmentation
    masks_label_fps.sort(key=lambda f: int(regex.sub('\D', '', f)))
    masks_label_fps = np.array([os.path.join(data_dir, image) for image in masks_label_fps])

    masks_is_fps = [img for img in listdir_nohidden(data_dir) if img.endswith(".is.png")]  # instance segmentation
    masks_is_fps.sort(key=lambda f: int(regex.sub('\D', '', f)))
    masks_is_fps = np.array([os.path.join(data_dir, image) for image in masks_is_fps])

    images_to_iterate_over = np.arange(len(images_fps))

    if TESTING_STAGE:
        images_to_iterate_over = np.array([0])  # only the first one for testing

    for image_id in tqdm(images_to_iterate_over):
        if TESTING_STAGE:
            print("adding image " + str(image_id) + " to json.")
        
        t0 = time.time()

        image = cv2.imread(images_fps[image_id])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_label = cv2.imread(masks_label_fps[image_id], 0)
        mask_is = cv2.imread(masks_is_fps[image_id], 0)
        if TESTING_STAGE:
            print("images_fps[i]", images_fps[image_id])
            print("masks_label_fps[i]", masks_label_fps[image_id])
            print("masks_is_fps[i]", masks_is_fps[image_id])

        image_info = pycococreatortools.create_image_info(
            image_id, os.path.basename(images_fps[image_id]), tuple(image.shape[:2]))
        coco_output["images"].append(image_info)

        mask_label_colours = np.unique(mask_label)
        mask_label = [(mask_label == v) for v in mask_label_colours]
        mask_label = np.stack(mask_label, axis=-1).astype('float')

        if TESTING_STAGE:
            print("mask_label", mask_label.shape)
            print("CATEGORIES len", len(CATEGORIES))
            if mask_label.shape[-1] > len(CATEGORIES):
                print("ERROR: There exist more masks on the class segmentation than there are categories!")

        mask_is_colours = np.unique(mask_is)
        mask_is = [(mask_is == v) for v in mask_is_colours]
        mask_is = np.stack(mask_is, axis=-1).astype('float')

        if TESTING_STAGE:
            print("mask_is_colours", mask_is_colours)
            print("mask_is", mask_is.shape)

        t1 = time.time()
        print("prep work: %f" % (t1 - t0))
        t0 = time.time()

        # we need to match the mask_label to the mask_is
        # is_labels gives the class of each instance segmented mask
        is_labels = np.full(mask_is.shape[-1], -1)  # array of -1s
        for is_id in np.arange(mask_is.shape[-1]):
            a_mask_is = mask_is[..., is_id]
            all_diff_counts = np.zeros(mask_label.shape[-1])
            for k in np.arange(mask_label.shape[-1]):
                a_mask_label = mask_label[..., k]
                bitwise_and = cv2.bitwise_and(a_mask_is, a_mask_label) # intersection of both masks
                diff = np.abs(bitwise_and - a_mask_is) # difference between intersection and instance segmentation mask
                count_non_zero = np.count_nonzero(diff)
                all_diff_counts[k] = count_non_zero

                if count_non_zero < 100:  # hardcoded margin of error!
                    is_labels[is_id] = k
                    break
            
            # if all the counts were over the highest margins, choose the class label with the smallest difference
            if is_labels[is_id] == -1:
                is_labels[is_id] = np.argmin(all_diff_counts)
        
        t1 = time.time()
        print("computation work: %f" % (t1 - t0))

        t0 = time.time()

        if TESTING_STAGE:
            print("is_labels", is_labels)
            # assume categories are ordered:
            class_names = [category['name'] for category in CATEGORIES]
            print("class_names", len(class_names), class_names)
            visualize_masks(image, mask_label, labels=class_names, title="label masks")
            print("mask_is.shape", mask_is.shape)
            is_names = [class_names[is_label] for is_label in is_labels]
            visualize_masks(image, mask_is, labels=is_names, title="instance segmentation masks")

        for is_id in np.arange(mask_is.shape[-1]):
            if START_AT_0:
                category_info = {'id': int(is_labels[is_id]), 'is_crowd': False}
            else:
                # the +1 here is because COCO wants the IDs to start at 1
                category_info = {'id': int(is_labels[is_id]) + 1, 'is_crowd': False}

            annotation_info = pycococreatortools.create_annotation_info(
                total_is_id, image_id, category_info, mask_is[..., is_id].squeeze(), tolerance=2)

            if annotation_info is not None:
                coco_output["annotations"].append(annotation_info)

            total_is_id += 1

        t1 = time.time()
        print("appending work: %f" % (t1 - t0))

    with open(JSON_PATH, 'w') as output_json_file:
        indent = None
        if TESTING_STAGE:
            indent = 4
            print("saving json.")

        json.dump(coco_output, output_json_file, indent=indent, sort_keys=True,
                  separators=(', ', ': '), ensure_ascii=False,
                  cls=NumpyEncoder)

    if TESTING_STAGE:
        plt.show()

if __name__ == "__main__":
    main()