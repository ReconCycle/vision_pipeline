import cv2
import numpy as np
import matplotlib.pyplot as plt
import json


def visualize_masks(image, masks, labels=None, title=""):
    n = masks.shape[-1]
    fig = plt.figure(figsize=(16, 5))
    if title is not "":
        fig.suptitle(title)
    for i in np.arange(-1, n):
        plt.subplot(1, n + 1, i + 2)
        plt.xticks([])
        plt.yticks([])
        name = str(i)
        plt.title(name)
        if i == -1:
            plt.imshow(image)
        else:
            plt.imshow(masks[..., i].squeeze())
            if labels is not None:
                plt.title(name + " " + labels[i])

    plt.show(block=False)

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show(block=False)

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)